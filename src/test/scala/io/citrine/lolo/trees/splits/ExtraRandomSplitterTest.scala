package io.citrine.lolo.trees.splits

import io.citrine.lolo.TestUtils
import org.junit.Test
import org.mockito.invocation.InvocationOnMock
import org.mockito.MockitoSugar

import scala.util.Random

class ExtraRandomSplitterTest extends MockitoSugar {

  /**
    * Test that uniform labels result in "NoSplit" with zero reduced impurity
    */
  @Test
  def testZeroVariance(): Unit = {
    val rng = new Random(278345L)
    val splitter = ExtraRandomSplitter(rng)
    val testData = Seq.fill(64){
      val x = rng.nextDouble()
      val y = 1.0
      val weight = 1.0
      (Vector(x), y, weight)
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
    val rng = new Random(278345L)
    val splitter = ExtraRandomSplitter(rng)
    val testData = Seq.fill(256){
      val x = rng.nextDouble()
      val y = rng.nextGaussian() * 1.0e-9 + 1.0
      val weight = 1.0
      (Vector(x), y, weight)
    }

    splitter.getBestSplit(testData, 1, 1)
  }

  /**
    * Test correctness of the split.
    */
  @Test
  def testCorrectSplit(): Unit = {
    val rng = new Random(34682L)
    val baseGrid = Seq(0.0, 1.2)
    Seq(1, 4, 8).foreach { numFeatures =>
      Seq(TestUtils.enumerateGrid(Seq.fill(numFeatures)(baseGrid)), Seq.fill(64)(Vector.fill(numFeatures)(rng.nextGaussian()))).foreach { xTrain =>
        // Repetitions with the same configuration.
        Seq(4).foreach { _ =>
          val featureIndices = (0 until numFeatures).toVector
          val shuffledFeatureIndices = rng.shuffle(featureIndices)

          // Mock random so that it produces the same shuffle() and nextDouble() sequence we achieved above.
          // It would be simpler to do this by resetting rng to the same initial seed. Using a mock, on the other hand, enables us to
          // detect when the test becomes invalid solely on the basis of functionally inconsequential refactoring that changes the
          // sequence of rng invocations.
          val mockRng = mock[Random]
          val randomUniforms = Seq.fill(numFeatures)(rng.nextDouble())
          doReturn(shuffledFeatureIndices).when(mockRng).shuffle(featureIndices)
          var nextDoubleInvocationCounter = 0
          when(mockRng.nextDouble) thenAnswer ((_: InvocationOnMock) => {
            nextDoubleInvocationCounter += 1
            randomUniforms(nextDoubleInvocationCounter - 1)
          })

          val featureCoefficients = rng.shuffle(featureIndices).map { i => Math.pow((i + 1).toDouble, 2.0) }
          val trainingData = xTrain.map { x =>
            val y = x.zip(featureCoefficients).map { case (a, b) => a * b }.sum
            val weight = 1.0
            (x, y, weight)
          }

          val cutPoints = shuffledFeatureIndices.zip(randomUniforms).map { case (i, u) =>
            val x =  trainingData.map(_._1(i))
            val xmin = x.min
            val xmax = x.max
            val cutPoint = xmin + u * (xmax - xmin)
            (i, cutPoint)
          }.sortBy(_._1).map(_._2)

          val varianceReductions = featureIndices.map { k =>
             trainingData.groupBy { v => v._1(k) < cutPoints(k) }.flatMap { case (_, subset) =>
              val mean = subset.map(_._2).sum / subset.length
              subset.map { case (_, yi, _) =>
                Math.pow(yi - mean, 2)
              }
            }.sum
          }

          val indexOfBest = shuffledFeatureIndices.take(numFeatures).map { i =>
            (i, varianceReductions(i))
          }.minBy(_._2)._1

          val splitter = ExtraRandomSplitter(mockRng)
          val bestSplit = splitter.getBestSplit( trainingData, numFeatures, 1)

          // Verifying invocation order helps detect when code modifications have invalidated this test.
          val order = inOrder(mockRng)
          order.verify(mockRng, times(1)).shuffle(featureIndices)
          order.verify(mockRng, times(numFeatures)).nextDouble()
          verifyNoMoreInteractions(mockRng)

          // Finally, we can ensure that the index on which we split is correct.
          assert(bestSplit._1.getIndex() == indexOfBest)
          // And ensure that we're turning at the right place.
          assert(bestSplit._1.turnLeft(Vector.fill(numFeatures)(xTrain.flatten.min)))
          assert(!bestSplit._1.turnLeft(Vector.fill(numFeatures)(xTrain.flatten.max)))
          assert(bestSplit._1.turnLeft(Vector.fill(numFeatures)(cutPoints(indexOfBest) - 1e-8)))
          assert(!bestSplit._1.turnLeft(Vector.fill(numFeatures)(cutPoints(indexOfBest) + 1e-8)))
        }
      }
    }
  }

}

