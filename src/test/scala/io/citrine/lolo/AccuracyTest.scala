package io.citrine.lolo

import java.io.FileWriter

import io.citrine.lolo.bags.{BaggedMultiResult, Bagger}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.transformers.FeatureRotator
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{BoltzmannSplitter, RegressionSplitter}
import io.citrine.theta.Stopwatch
import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 7/10/17.
  */
class AccuracyTest {

  val noiseLevel: Double = 0.000
  val nFeat: Int = 10
  val nRow: Int = 128
  val nScal: Int = 2
  val minInstances: Int = 1

  val trainingData: Seq[(Vector[Any], Double)]  = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(nRow, nFeat, noise = noiseLevel, seed = 3L),
    inputBins = Seq((4, 32))
  ).asInstanceOf[Seq[(Vector[Any], Double)]] // binTrainingData isn't binning the labels

  // Get the out-of-bag RMSE
  private def computeMetrics(learner: Learner): Double = {
    learner.train(trainingData).getLoss().get
  }

  /**
    * Quick sanity check of the test setup
    */
  @Test
  def testRandomForest(): Unit = {
    val baseLearner = RegressionTreeLearner(numFeatures = nFeat / 3, minLeafInstances = minInstances)
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
        splitter = RegressionSplitter(randomizePivotLocation = true)
      )
      val learner = new Bagger(baseLearner, numBags = nRow * 16)
      // println(s"Normal train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner)
    }
    val errorAnnealingTree = {
      val baseLearner = RegressionTreeLearner(
        numFeatures = nFeat,
        splitter = BoltzmannSplitter(temperature = Float.MinPositiveValue)
      )
      val learner = new Bagger(baseLearner, numBags = nRow * 16)
      // println(s"Annealing train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner)
    }

    val relativeDifference = 2.0 * Math.abs(errorAnnealingTree - errorStandardTree)/(errorAnnealingTree + errorStandardTree)
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
object AccuracyTest {

  val trainingDataFull: Seq[(Vector[Any], Double)] = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(2048, 48),
    inputBins = Seq((2, 32)) // bin the 3rd feature into a categorical
  ).asInstanceOf[Seq[(Vector[Any], Double)]]

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
  def computeMetrics(nRow: Int, nFeat: Int, nFeatSub: Int, nScal: Int, minInstances: Int, temperature: Double): (Double, Double) = {
    val trainingData = trainingDataFull.take(nRow).map{case (f, l) => (f.take(nFeat), l)}
    val splitter = if (temperature > 0) {
      BoltzmannSplitter(temperature = temperature)
    } else {
      RegressionSplitter(randomizePivotLocation = true)
    }
    val baseLearner = RegressionTreeLearner(
      numFeatures = nFeatSub,
      splitter = splitter,
      minLeafInstances = minInstances
    )
    val learner = new Bagger(baseLearner, numBags = nRow * nScal, biasLearner = None)
    val model = learner.train(trainingData).getModel()
    val (features, labels) = trainingData.unzip
    val predictions = model.transform(features)
    val expected = predictions.getExpected().asInstanceOf[Seq[Double]]
    val sigma = predictions.getUncertainty().get.asInstanceOf[Seq[Double]]
    val pva: Seq[(Double, Double, Double)] = labels.indices.map{i =>
      (labels(i), expected(i), sigma(i))
    }
    val rmse = Math.sqrt(pva.map{case (x, y, s) => Math.pow(x-y, 2)}.sum / pva.size)
    val stdres = Math.sqrt(pva.map{case (x, y, s) => Math.pow((x-y) / s, 2)}.sum / pva.size)
    (rmse, stdres)
  }

  def randomRotationTest(
                         function: Seq[Double] => Double = Friedman.friedmanSilverman,
                         outputFilename: String = s"random_rotation_test.csv",
                         append: Boolean = false
                        ) = {
    val nTest = 64
    val rand = new Random()
    val splitter = RegressionSplitter(randomizePivotLocation = false)
    val plainBaseLearner= RegressionTreeLearner(splitter = splitter)
    val rotatedBaseLearner = FeatureRotator(plainBaseLearner)

    {
      val fw = new FileWriter(outputFilename)
      try {
        fw.write(f"nRow,trainingNoise,testNoise,numBags,useJackknife,randomRotation,rmse,stdconf,ucorr,stdres\n")
      } finally fw.close()
    }

    Seq(32, 128, 512).foreach { nRow =>
      val trainSeed = rand.nextLong()
      val testSeed = rand.nextLong()
      Seq(0.0, 0.25, 0.5, 1.0, 2.0, 4.0).foreach { trainingNoise =>
        val trainingData = TestUtils.generateTrainingData(nRow, 10, function = function, seed = trainSeed, noise = trainingNoise)
        Seq(32, 128, 512).foreach { numBags =>
          Seq(0.0, trainingNoise).foreach { testNoise =>
            Seq(true, false).foreach { randomRotation =>
              val baseLearner = if (randomRotation) rotatedBaseLearner else plainBaseLearner
              val learner = new Bagger(baseLearner, numBags = numBags, useJackknife = true, uncertaintyCalibration = true, biasLearner = None)
              val model = learner.train(trainingData).getModel()
              val testData = TestUtils.generateTrainingData(nTest, 10, function = function, seed = testSeed, noise = testNoise)
              val (testFeatures, testLabels) = testData.unzip

              val predictions: BaggedMultiResult with PredictionResult[Double] = model.transform(testFeatures).asInstanceOf[BaggedMultiResult with PredictionResult[Double]]
              val stdconf = validation.StandardConfidence.evaluate(predictions, testLabels)
              val ucorr = validation.UncertaintyCorrelation.evaluate(predictions, testLabels)
              val stdres = validation.StandardError().evaluate(predictions, testLabels)
              val rmse = validation.RootMeanSquareError.evaluate(predictions, testLabels)

              val fw = new FileWriter(outputFilename, true)
              try {
                fw.write(f"$nRow,$trainingNoise,$testNoise,${learner.numBags},${predictions.useJackknife},$randomRotation,${rmse}%7.5f,${stdconf}%7.5f,${ucorr}%7.5f,${stdres}%7.5f\n")
              } finally fw.close()
            }
          }
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {
    randomRotationTest()
    return

    var temp = 0.00001
    val nRow = 256
    val nFeat = 48
    val nScal = 1
    val minInstances = 1
    val (baseRMSE: Double, baseStdRes: Double) = computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temperature = 0.0)
    println(baseRMSE, baseStdRes)
    val improvements: Seq[(Double, Double, Double)] = (0 until 64).map { i =>
      val (error, stdres) = computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temp)
      println(f"Error in annealing tree at T=${temp}%6.2e is ${error}%5.2f (${error / baseRMSE}%5.3f, ${stdres}%5.3f)")
      val res = (temp, baseRMSE / error, baseStdRes / stdres)
      temp = temp * Math.pow(2.0, 1.0/4.0)
      res
    }
    val best = improvements.maxBy(_._2)
    println(f"The best improvement was ${best._2}%5.2f x at T=${best._1}%6.2e")
    val bestt = improvements.maxBy(_._3)
    println(f"The best improvement was ${bestt._3}%5.2f x at T=${bestt._1}%6.2e")
  }
}
