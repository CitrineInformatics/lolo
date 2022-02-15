package io.citrine.lolo

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{BoltzmannSplitter, RegressionSplitter}
import org.junit.Test

/**
  * Created by maxhutch on 7/10/17.
  */
class AccuracyTest extends SeedRandomMixIn {

  val noiseLevel: Double = 0.000
  val nFeat: Int = 10
  val nRow: Int = 128
  val nScal: Int = 2
  val minInstances: Int = 1

  val trainingData: Seq[(Vector[Any], Double)] = TestUtils
    .binTrainingData(
      TestUtils.generateTrainingData(nRow, nFeat, noise = noiseLevel, rng = rng),
      inputBins = Seq((4, 32))
    )
    .asInstanceOf[Seq[(Vector[Any], Double)]] // binTrainingData isn't binning the labels

  // Get the out-of-bag RMSE
  private def computeMetrics(learner: Learner): Double = {
    learner.train(trainingData).getLoss().get
  }

  /**
    * Test that setting rng yields repeatable results.
    */
  /**
    * DISABLED until implementing perfect reproducibility.
    *  @Test
    *  def testRepeatable(): Unit = {
    *    val (errorsStandardTree, errorsAnnealingTree, errorsUnrandomizedTree) = (1 to 4).map { _ =>
    *      val seed = 354128L
    *      val rng = new Random(seed)
    *      val errorStandardTree = {
    *        val baseLearner = RegressionTreeLearner(
    *          numFeatures = nFeat,
    *          splitter = RegressionSplitter(randomizePivotLocation = true, rng = rng),
    *          rng = rng
    *        )
    *        val learner = new Bagger(baseLearner, numBags = nRow * 8, randBasis = TestUtils.getBreezeRandBasis(rng.nextLong()))
    *        computeMetrics(learner)
    *      }
    *      rng.setSeed(seed)
    *      val errorAnnealingTree = {
    *        val baseLearner = RegressionTreeLearner(
    *          numFeatures = nFeat,
    *          splitter = BoltzmannSplitter(temperature = Float.MinPositiveValue, rng = rng),
    *          rng = rng
    *        )
    *        val learner = new Bagger(baseLearner, numBags = nRow * 8, randBasis = TestUtils.getBreezeRandBasis(rng.nextLong()))
    *        computeMetrics(learner)
    *      }
    *      rng.setSeed(seed)
    *      val randBasis = TestUtils.getBreezeRandBasis(rng.nextLong())
    *      val errorUnrandomizedTree = {
    *        val baseLearner = RegressionTreeLearner(
    *          numFeatures = nFeat,
    *          splitter = RegressionSplitter(randomizePivotLocation = false, rng = rng),
    *          rng = rng
    *        )
    *        val learner = new Bagger(baseLearner, numBags = nRow * 8, randBasis = randBasis)
    *        computeMetrics(learner)
    *      }
    *      (errorStandardTree, errorAnnealingTree, errorUnrandomizedTree)
    *    }.unzip3
    *
    *    val atol = 1e-12
    *    assert(errorsStandardTree.forall{ e => Math.abs(e - errorsStandardTree.head) < atol })
    *    assert(errorsAnnealingTree.forall{ e => Math.abs(e - errorsStandardTree.head) < atol })
    *    assert(errorsUnrandomizedTree.forall{ e => Math.abs(e - errorsUnrandomizedTree.head) < atol })
    *  }
    */

  /**
    * Quick sanity check of the test setup
    */
  @Test
  def testRandomForest(): Unit = {
    val baseLearner = RegressionTreeLearner(numFeatures = nFeat / 3, minLeafInstances = minInstances, rng = rng)
    val learner = new Bagger(baseLearner, numBags = nRow * nScal)
    val error = computeMetrics(learner)
    assert(error > noiseLevel, s"Can't do better than noise")
    assert(error < 4.0, "Error increased, probably due to a change in configuration")
  }

  /**
    * Check that a low-temperature Boltzmann Tree recovers the performance of a normal RF (with randomized pivots)
    */
  @Test
  def testLowTLimit(): Unit = {
    val errorStandardTree = {
      val baseLearner = RegressionTreeLearner(
        numFeatures = nFeat,
        splitter = RegressionSplitter(randomizePivotLocation = true, rng = rng),
        rng = rng
      )
      val learner = new Bagger(baseLearner, numBags = nRow * 16, randBasis = TestUtils.breezeRandBasis(rng.nextLong()))
      // println(s"Normal train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner)
    }
    val errorAnnealingTree = {
      val baseLearner = RegressionTreeLearner(
        numFeatures = nFeat,
        splitter = BoltzmannSplitter(temperature = Float.MinPositiveValue, rng = rng),
        rng = rng
      )
      val learner = new Bagger(baseLearner, numBags = nRow * 16, randBasis = TestUtils.breezeRandBasis(rng.nextLong()))
      // println(s"Annealing train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner)
    }

    val relativeDifference =
      2.0 * Math.abs(errorAnnealingTree - errorStandardTree) / (errorAnnealingTree + errorStandardTree)
    // println(relativeDifference)
    assert(relativeDifference < 0.01)
  }
}

/**
  * Driver code to study the performance vs temperature
  *
  * This isn't cast as a test, but can be used to try to understand the behavior of Boltzmann trees on some simple problems.
  * TODO: turn this into a demo or otherwise relocate it before the Boltzmann tree release
  */
object AccuracyTest extends SeedRandomMixIn {

  val trainingDataFull: Seq[(Vector[Any], Double)] = TestUtils
    .binTrainingData(
      TestUtils.generateTrainingData(2048, 48, rng = rng),
      inputBins = Seq((2, 32)) // bin the 3rd feature into a categorical
    )
    .asInstanceOf[Seq[(Vector[Any], Double)]]

  /**
    * Compute the RMSE and standard residual for a Boltzmann tree with the given temperature
    * @param nRow training set size
    * @param nFeat number of features
    * @param nFeatSub number of features considered per split
    * @param nScal number of trees, as a multiple of the training set size
    * @param minInstances minimum tree node size
    * @param temperature of the boltzmann tree
    * @return (RMSE, standard residual)
    */
  def computeMetrics(
      nRow: Int,
      nFeat: Int,
      nFeatSub: Int,
      nScal: Int,
      minInstances: Int,
      temperature: Double
  ): (Double, Double) = {
    val trainingData = trainingDataFull.take(nRow).map { case (f, l) => (f.take(nFeat), l) }
    val splitter = if (temperature > 0) {
      BoltzmannSplitter(temperature = temperature)
    } else {
      RegressionSplitter(randomizePivotLocation = true)
    }
    val baseLearner = RegressionTreeLearner(
      numFeatures = nFeatSub,
      splitter = splitter,
      minLeafInstances = minInstances,
      rng = rng
    )
    val learner = new Bagger(baseLearner, numBags = nRow * nScal, biasLearner = None)
    val model = learner.train(trainingData).getModel()
    val (features, labels) = trainingData.unzip
    val predictions = model.transform(features)
    val expected = predictions.getExpected().asInstanceOf[Seq[Double]]
    val sigma = predictions.getUncertainty().get.asInstanceOf[Seq[Double]]
    val pva: Seq[(Double, Double, Double)] = labels.indices.map { i =>
      (labels(i), expected(i), sigma(i))
    }
    val rmse = Math.sqrt(pva.map { case (x, y, s) => Math.pow(x - y, 2) }.sum / pva.size)
    val stdres = Math.sqrt(pva.map { case (x, y, s) => Math.pow((x - y) / s, 2) }.sum / pva.size)
    (rmse, stdres)
  }

  def main(args: Array[String]): Unit = {
    var temp = 0.00001
    val nRow = 256
    val nFeat = 48
    val nScal = 1
    val minInstances = 1
    val (baseRMSE: Double, baseStdRes: Double) =
      computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temperature = 0.0)
    println(baseRMSE, baseStdRes)
    val improvements: Seq[(Double, Double, Double)] = (0 until 64).map { i =>
      val (error, stdres) = computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temp)
      println(f"Error in annealing tree at T=${temp}%6.2e is ${error}%5.2f (${error / baseRMSE}%5.3f, ${stdres}%5.3f)")
      val res = (temp, baseRMSE / error, baseStdRes / stdres)
      temp = temp * Math.pow(2.0, 1.0 / 4.0)
      res
    }
    val best = improvements.maxBy(_._2)
    println(f"The best improvement was ${best._2}%5.2f x at T=${best._1}%6.2e")
    val bestt = improvements.maxBy(_._3)
    println(f"The best improvement was ${bestt._3}%5.2f x at T=${bestt._1}%6.2e")
  }
}
