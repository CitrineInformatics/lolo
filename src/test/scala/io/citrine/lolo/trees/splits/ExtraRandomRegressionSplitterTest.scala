package io.citrine.lolo.trees.splits

import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import io.citrine.random.Random
import org.junit.Test

class ExtraRandomRegressionSplitterTest extends SeedRandomMixIn {

  /**
    * Test that uniform labels result in "NoSplit" with zero reduced impurity
    */
  @Test
  def testZeroVariance(): Unit = {
    val splitter = ExtraRandomRegressionSplitter()
    val testData = Seq.fill(64) {
      val x = rng.nextDouble()
      val y = 1.0
      val weight = 1.0
      TrainingRow(Vector(x), y, weight)
    }

    val (bestSplit, improvement) = splitter.getBestSplit(testData, 1, 1)
    assert(bestSplit.isInstanceOf[NoSplit], "Can't split on zero variance")
    assert(improvement == 0, "Can't improve on no variance")
  }

  /**
    * Test that very small random signals don't lead to exceptions.
    *
    * Note that this test will return NoSplit occasionally, when the starting impurity is numerically zero.  That's OK
    * though.
    */
  @Test
  def testLowVariance(): Unit = {
    val splitter = ExtraRandomRegressionSplitter()
    val testData = Seq.fill(256) {
      val x = rng.nextDouble()
      val y = rng.nextGaussian() * 1.0e-9 + 1.0
      val weight = 1.0
      TrainingRow(Vector(x), y, weight)
    }

    splitter.getBestSplit(testData, 1, 1, rng = rng)
  }

  /**
    * Test correctness of the split.
    *
    * Specifically, this test ensures that ExtraRandomSplitter.getBestSplit selects the split
    */
  @Test
  def testCorrectSplit(): Unit = {
    val baseGrid = Seq(0.0, 1.2)
    Seq(1, 5, 9).foreach { numFeatures =>
      // Arrange training input data as a regular grid, and then as a Gaussian-distributed sample.
      Seq(
        DataGenerator.enumerateGrid(Seq.fill(numFeatures)(baseGrid)),
        Seq.fill(64)(Vector.fill(numFeatures)(rng.nextGaussian()))
      ).foreach { xTrain =>
        // Have ExtraRandomSplitter.getBestSplit choose a split from this number of randomly-chosen features.
        (1 to numFeatures by Math.max(numFeatures / 2, 1)).foreach { numFeaturesToConsider =>
          // Repeat with the same training inputs a few times with different random number streams.
          (1 to 4).foreach { repetitionNumber =>
            // Feature indices, which will be used to determine which features will be considered by the splitter (and in which order).
            // This is stored as a Vector because rng.shuffle was having trouble with a Seq; possibly a type erasure issue.
            val featureIndices = (0 until numFeatures).toVector

            // Generate linear training data, using xTrain from above as inputs, starting by selecting coefficients for each feature.
            val featureCoefficients = rng.shuffle(featureIndices).map { i => Math.pow((i + 1).toDouble, 2.0) }
            val trainingTuples = xTrain.map { x =>
              val y = x.zip(featureCoefficients).map { case (a, b) => a * b }.sum
              val weight = 1.0
              (x, y, weight)
            }

            // Determine exactly what shuffle() and nextDouble() will return within getBestSplit by resetting rng to a common rng seed.
            val sharedSeed = 238745L + repetitionNumber
            rng = Random(sharedSeed)
            // This is the set of shuffled indices to return when getBestSplit calls the RNG's shuffle method.
            val shuffledFeatureIndices = rng.shuffle(featureIndices)
            // Precompute the sequence of numbers returned when getBestSplit calls the RNG's nextDouble method to select cut points.
            val randomUniforms = Seq.fill(numFeatures)(rng.nextDouble())

            // Compute where the cut points should be placed, based on the the sequence of randomUniforms.
            val cutPoints = shuffledFeatureIndices
              .zip(randomUniforms)
              .map {
                case (i, u) =>
                  val x = trainingTuples.map(_._1(i))
                  val xmin = x.min
                  val xmax = x.max
                  val cutPoint = xmin + u * (xmax - xmin)
                  (i, cutPoint)
              }
              .sortBy(_._1)
              .map(_._2)

            // Compute the sum of variances across partitions of each possible cut.
            val varianceSums = featureIndices.map { k =>
              trainingTuples
                .groupBy { v => v._1(k) < cutPoints(k) }
                .flatMap {
                  case (_, subset) =>
                    val mean = subset.map(_._2).sum / subset.length
                    subset.map {
                      case (_, yi, _) =>
                        Math.pow(yi - mean, 2)
                    }
                }
                .sum
            }

            // Choose the index on which a split results in the lowest sum of variances.
            val indexOfBest = shuffledFeatureIndices
              .take(numFeaturesToConsider)
              .map { i =>
                (i, varianceSums(i))
              }
              .minBy(_._2)
              ._1

            // Instantiate the splitter to test, passing in the random number generator that is reset to its former state used above.
            rng = Random(sharedSeed)
            val splitter = ExtraRandomRegressionSplitter()
            // Ask the splitter what it chose as a split.
            val bestSplit =
              splitter.getBestSplit(TrainingRow.build(trainingTuples), numFeaturesToConsider, 1, rng = rng)

            // Finally, we can ensure that the index on which we split is correct...
            val testCaveatMessage = "NOTE: this test may inaccurately fail due to changes in the sequence of rng calls."
            assert(bestSplit._1.index == indexOfBest, s"Incorrect index of best split. $testCaveatMessage")
            // Do a sanity check about the directionality of turnLeft to ensure this test is valid.
            assert(
              bestSplit._1.turnLeft(Vector.fill(numFeatures)(xTrain.flatten.min)),
              s"Unexpected directionality of turnLeft, which is probably a bug with the test itself."
            )
            assert(
              !bestSplit._1.turnLeft(Vector.fill(numFeatures)(xTrain.flatten.max)),
              s"Unexpected directionality of turnLeft, which is probably a bug with the test itself."
            )
            // And ensure that we're turning at the right place.
            assert(
              bestSplit._1.turnLeft(Vector.fill(numFeatures)(cutPoints(indexOfBest) - 1e-8)),
              s"Split not placed at correct location. $testCaveatMessage"
            )
            assert(
              !bestSplit._1.turnLeft(Vector.fill(numFeatures)(cutPoints(indexOfBest) + 1e-8)),
              s"Split not placed at correct location. $testCaveatMessage"
            )
          }
        }
      }
    }
  }
}
