package io.citrine.lolo

import java.io.FileWriter

import io.citrine.lolo.bags.{BaggedMultiResult, Bagger}
import io.citrine.lolo.stats.functions.{Cute, Friedman, Linear}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{BoltzmannSplitter, ExtraRandomSplitter, RegressionSplitter}
import io.citrine.lolo.validation.UncertaintyCorrelation
import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 7/10/17.
  */
class AccuracyTest {

  val noiseLevel: Double = 0.000
  val nFeat: Int = 10
  val nRow: Int = 128
  val nTree: Int = 2
  val minInstances: Int = 1

  val trainingData: Seq[(Vector[Any], Double)] = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(nRow, nFeat, noise = noiseLevel, seed = 3L),
    inputBins = Seq((4, 32))
  ).asInstanceOf[Seq[(Vector[Any], Double)]] // binTreeingData isn't binning the labels

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
    val learner = new Bagger(baseLearner, numBags = nRow * nTree)
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

    val relativeDifference = 2.0 * Math.abs(errorAnnealingTree - errorStandardTree) / (errorAnnealingTree + errorStandardTree)
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


  /**
    * Compute the RMSE and standard residual for a Boltzmann tree with the given temperature
    *
    * @param nFeatSub     number of features considered per split
    * @param nTree        number of trees, as a multiple of the training set size
    * @param minInstances minimum tree node size
    * @param temperature  of the boltzmann tree
    * @return (RMSE, standard residual)
    */
  def computeMetrics(
                      name: String,
                      trainingData: Seq[(Vector[Any], Double)],
                      testData: Seq[(Vector[Any], Double)],
                      nFeatSub: Int, nTree: Int, minInstances: Int,
                      temperature: Double, extra: Boolean, jackknife: Boolean = false, relative: Boolean = true, bagging: Boolean = true
                    ): (Double, Double, Double, Double, Double) = {
    val nRow = trainingData.size

    val (testFeatures, testLabels) = testData.unzip
    val splitter = if (extra) {
      ExtraRandomSplitter()
    } else if (temperature > 0) {
      val trainMean = trainingData.map(_._2).sum / trainingData.size
      val trainTotalVariance = trainingData.map(x => Math.pow(x._2 - trainMean, 2.0)).sum
      if (relative) {
        BoltzmannSplitter(temperature = temperature, relative = relative)
      } else {
        BoltzmannSplitter(temperature = temperature * trainTotalVariance, relative = relative)
      }
    } else {
      RegressionSplitter()
    }
    val baseLearner = RegressionTreeLearner(
      numFeatures = nFeatSub,
      splitter = splitter,
      minLeafInstances = minInstances
    )
    val learner = new Bagger(baseLearner, numBags = nTree, useJackknife = jackknife, useBagging = bagging)
    val model = learner.train(trainingData).getModel()
    val predictions: BaggedMultiResult with PredictionResult[Double] = model.transform(testFeatures).asInstanceOf[BaggedMultiResult with PredictionResult[Double]]

    val rmse = validation.RootMeanSquareError.evaluate(predictions, testLabels)
    val ucorr = validation.UncertaintyCorrelation.evaluate(predictions, testLabels)
    val corr = predictions.getCorrelation(testLabels)
    val pCor = predictions.getPredictiveCorrelation()
    val pvar = predictions.getPredictedVariance()
    val tError = predictions.getTreeError(testLabels)

    val fname = s"data_nrow-${nRow}_${name}.csv"
    val fw = new FileWriter(fname, true)
    try {
      val name = if (extra) {
        s"extra-${nFeatSub}"
      } else if (temperature == 0) {
        s"cart-${nFeatSub}"
      } else if (relative) {
        f"boltz-${temperature}%6.2e"
      } else {
        f"boltz2-${temperature}%6.2e"
      }
      fw.write(f"$name,$extra,$relative,$bagging, $nFeatSub, ${temperature}%6.2e, ${rmse * rmse}%7.5f, ${corr}%7.5f, ${pCor - 1}%7.5f, ${tError}%7.5f\n")
    } finally fw.close()

    (rmse * rmse, ucorr, pCor, corr, tError)
  }

  def runAll(
              trainingData: Seq[(Vector[Any], Double)],
              testData: Seq[(Vector[Any], Double)],
              name: String,
              nTree: Int
            ): Unit = {
    val minInstances = 1

    val (testFeatures, testLabels) = testData.unzip
    val mean = testLabels.sum / testLabels.size
    // println(s"${trainingData.size},  GTM = ${Math.sqrt(testLabels.map(x => Math.pow(x - mean, 2.0)).sum / testLabels.size)}")

    val nFeat = trainingData.head._1.size

    (1 to nFeat).foreach { k =>
      println(computeMetrics(name, trainingData, testData, k, nTree, minInstances, temperature = 0.0, false))
    }

    (1 to nFeat).foreach { k =>
      println(computeMetrics(name, trainingData, testData, k, nTree, minInstances, 0.0, true, bagging = false))
    }

    (1 to nFeat).foreach { k =>
      println(computeMetrics(name, trainingData, testData, k, nTree, minInstances, 0.0, true, bagging = true))
    }

    var temp = 0.01
    (0 until 16).map { i =>
      println(computeMetrics(name, trainingData, testData, nFeat, nTree, minInstances, temp, false))
      temp = temp * Math.pow(2.0, 1.0 / 2.0)
      (temp, temp, temp)
    }

    temp = 0.01
    (0 until 16).map { i =>
      println(computeMetrics(name, trainingData, testData, nFeat, nTree, minInstances, temp, false, bagging = false))
      temp = temp * Math.pow(2.0, 1.0 / 2.0)
      (temp, temp, temp)
    }
  }


  def testJackknife(func: Seq[Double] => Double = Friedman.friedmanSilverman): Unit = {
    val nTest = 512
    val rand = new Random()

    val splitter = RegressionSplitter(randomizePivotLocation = true)

    val baseLearner = RegressionTreeLearner(
      splitter = splitter
    )

    val biasLearner = new Bagger(baseLearner, numBags = 128, uncertaintyCalibration = false)

    Seq(16, 32, 64, 128, 256, 512, 1024).foreach { nRow =>
      val trainSeed = rand.nextLong()
      val testSeed = rand.nextLong()

      Seq(0.0, 0.125, 0.25, 0.5, 1.0).foreach { trainingNoise =>
        val trainingData = TestUtils.generateTrainingData(nRow, 10, function = func, seed = trainSeed, noise = trainingNoise, heteroscedastic = false)

        Seq(true).map { useJackknife =>
          Seq(32, 64, 128, 256, 512, 1024, 2048, 4096).foreach { nBag =>
            val learner = new Bagger(baseLearner, numBags = nBag, useJackknife = useJackknife, uncertaintyCalibration = true, biasLearner = None) // Some(biasLearner))
            val model = learner.train(trainingData).getModel()
            Seq(0.0, trainingNoise).foreach { testNoise =>
              val testData = TestUtils.generateTrainingData(nTest, 10, function = func, seed = testSeed, noise = testNoise, heteroscedastic = false)
              val (testFeatures, testLabels) = testData.unzip

              val fname = s"foo.csv"

              val orig: BaggedMultiResult with PredictionResult[Double] = model.transform(testFeatures).asInstanceOf[BaggedMultiResult with PredictionResult[Double]]

              if (true) {
                val predictions = orig
                val stdconf = validation.StandardConfidence.evaluate(predictions, testLabels)
                val ucorr = validation.UncertaintyCorrelation.evaluate(predictions, testLabels)
                val stdres = validation.StandardError().evaluate(predictions, testLabels)
                val rmse = validation.RootMeanSquareError.evaluate(predictions, testLabels)

                val fw = new FileWriter(fname, true)
                try {
                  fw.write(f"$nRow,$trainingNoise,$testNoise,${learner.numBags},${predictions.useJackknife},${true},${rmse}%7.5f,${stdconf}%7.5f,${ucorr}%7.5f,${stdres}%7.5f\n")
                } finally fw.close()
              }

              if (false) {
                val predictions = orig.copy(useJackknife = false).asInstanceOf[BaggedMultiResult with PredictionResult[Double]]
                val stdconf = validation.StandardConfidence.evaluate(predictions, testLabels)
                val ucorr = validation.UncertaintyCorrelation.evaluate(predictions, testLabels)
                val stdres = validation.StandardError().evaluate(predictions, testLabels)
                val rmse = validation.RootMeanSquareError.evaluate(predictions, testLabels)

                val fw = new FileWriter(fname, true)
                try {
                  fw.write(f"$nRow,$trainingNoise,$testNoise,${learner.numBags},${predictions.useJackknife},${true},${rmse}%7.5f,${stdconf}%7.5f,${ucorr}%7.5f,${stdres}%7.5f\n")
                } finally fw.close()
              }
            }
          }
        }
      }
    }
  }

  def testErrorDecomposition(func: Seq[Double] => Double = Friedman.friedmanSilverman): Unit = {
    val rand = new Random()
    val splitter = RegressionSplitter(randomizePivotLocation = true)

    val baseLearner = RegressionTreeLearner(
      splitter = splitter
    )

    while (true) {
      val testSeed: Long = rand.nextLong()
      val nTest = 256
      val nRep = 64
      val testData = TestUtils.generateTrainingData(nTest, 10, function = func, seed = testSeed, noise = 0.0, heteroscedastic = false)
      val (testFeatures, testLabels) = testData.unzip
      val testNoise = 0.0

      Seq(16, 32, 64, 128, 256, 512, 1024).foreach { nRow =>

        Seq(0.0, 1.0).foreach { trainingNoise =>

          Seq(1024).foreach{ nBag =>

            val learner = new Bagger(baseLearner, numBags = nBag, useJackknife = true, uncertaintyCalibration = false)

            val data = Seq.fill(nRep) {
              val trainSeed = rand.nextLong()
              val trainingData = TestUtils.generateTrainingData(nRow, 10, function = func, seed = trainSeed, noise = trainingNoise, heteroscedastic = false)

              val model = learner.train(trainingData).getModel()
              val res = model.transform(testFeatures).asInstanceOf[BaggedMultiResult]
              (res.getExpected().asInstanceOf[Seq[Double]], res.getUncertainty().get.asInstanceOf[Seq[Double]], res.getUncertainty().get.asInstanceOf[Seq[Double]]) // res.getBiasEstimate())
            }
            val allExpected: Seq[Seq[Double]] = data.map(_._1)
            val allUncertainty: Seq[Seq[Double]] = data.map(_._2)
            val allBias: Seq[Seq[Double]] = data.map(_._3)

            val mean: Seq[Double] = allExpected.transpose.map(x => x.sum / x.size)
            val bias: Seq[Double] = mean.zip(testLabels).map{case (y, bar) => Math.pow(y - bar, 2.0)}
            val variance: Seq[Double] = allExpected.transpose.zip(mean).map{case (x, avg) => x.map(y => Math.pow(y - avg, 2.0)).sum / x.size}

            val avgInterval = allUncertainty.transpose.map(x => x.map(Math.pow(_, 2.0)).sum / x.size)
            val intervalVariance = allUncertainty.transpose.zip(avgInterval).map{case (x, avg) => x.map(y => Math.pow(y - avg, 2.0)).sum / x.size}

            val rawError = allExpected.transpose.zip(testLabels).flatMap{case (x, y) => x.map(z => Math.pow(z - y, 2.0))}.toVector
            val rawVar = allUncertainty.transpose.flatten.toVector.map(Math.pow(_, 2))
            val allErr = rawError // rawError.zip(rawVar).map(x => Math.max(x._1 - x._2, 0.0)) //  Seq.fill(allExpected.size)(bias).transpose.flatten.toVector // allExpected.transpose.zip(mean).flatMap{case (x, y) => x.map(z => Math.pow(z - y, 2.0))}.toVector //
            val allUnc = rawVar // allBias.transpose.flatten.toVector.map(Math.pow(_, 2))
            // bias.zip(allUncertainty.transpose).map{case (b, x) => (b, x.sum / x.size, Math.sqrt(x.map(y => Math.pow(y, 2)).sum / x.size), x)}.foreach{println}

            val rmse = Math.sqrt(allExpected.transpose.zip(testLabels).flatMap{case (x, y) => x.map(z => Math.pow(z - y, 2.0))}.sum / (nRep * testLabels.size)) // Math.sqrt(bias.zip(variance).map{case (b, v) => b + v}.sum / bias.size)
            val varError = Math.sqrt(allExpected.transpose.zip(mean).flatMap{case (x, y) => x.map(z => Math.pow(z - y, 2.0))}.sum / (nRep * testLabels.size)) // Math.sqrt(bias.zip(variance).map{case (b, v) => b + v}.sum / bias.size)
            val stdconf = allErr.zip(allUnc).count(x => x._1 <= x._2).toDouble / allErr.size
            val ucorr = UncertaintyCorrelation.evaluate(
              new PredictionResult[Double] {
                /**
                  * Get the expected values for this prediction
                  *
                  * @return expected value of each prediction
                  */
                override def getExpected(): Seq[Double] = allErr.map(Math.sqrt)

                override def getUncertainty(): Option[Seq[Any]] = Some(allUnc.map(Math.sqrt))
              },
              Seq.fill(allErr.size)(0.0)
            )
            val stdres = Math.sqrt(allErr.zip(allUnc).map(x => x._1 / x._2).sum / allErr.size)

            val msbias = bias.sum / bias.size
            val msvariance = variance.sum / variance.size
            val intVarVar = intervalVariance.sum / intervalVariance.size

            val fname = s"decomp.csv"
            val fw = new FileWriter(fname, true)
            try {
              val line = f"$nRow,$trainingNoise,$testNoise,${learner.numBags},${learner.useJackknife},${learner.uncertaintyCalibration},${rmse}%7.5f,${msbias}%7.5f,${msvariance}%7.5f,${stdconf}%7.5f,${ucorr}%7.5f,${stdres}%7.5f,${varError}%7.5f"
              println(line)
              fw.write(s"$line\n")
            } finally fw.close()
          }
        }
      }
    }
  }


  def main(args: Array[String]): Unit = {
    val nTrial = 64
    val nTest = 1024
    val nTree = 64
    val rand = new Random(seed = 0L)

    val linear = Linear.randomDirection(8)

    while (true) {
      testJackknife()
      // testJackknife(linear.apply)
      // testJackknife(Linear(Seq.fill(8)(0.0)).apply)
      // testErrorDecomposition()
    }
    return

    (0 until nTrial).foreach { _ =>
      println("MSE\tUCorr\tPVar\tRho\tTE\n")
      if (false) {
        Seq(4, 6, 8, 12, 16).foreach { nEdge =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateGridData(Seq(nEdge, nEdge), function = Cute.fhi, xoff = -0.5, seed = rand.nextLong())
          val testData = TestUtils.generateTrainingData(nTest, 2, function = Cute.fhi, xoff = -0.5, seed = rand.nextLong())

          runAll(trainingData, testData, s"fhi-grid-${nEdge}", nTree)
        }
      }

      if (false) {
        Seq(4, 6, 8, 12, 16).foreach { nEdge =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateTrainingData(nEdge * nEdge, 2, function = Cute.fhi, xoff = -0.5, seed = rand.nextLong())
          val testData = TestUtils.generateTrainingData(nTest, 2, function = Cute.fhi, xoff = -0.5, seed = rand.nextLong())

          runAll(trainingData, testData, s"fhi-rand-$nEdge", nTree)
        }
      }

      if (true) {
        Seq(16, 32, 64, 128, 256, 512).foreach { nRow =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateTrainingData(nRow, 10, function = Friedman.friedmanSilverman, seed = rand.nextLong())
          val testData = TestUtils.generateTrainingData(nTest, 10, function = Friedman.friedmanSilverman, seed = rand.nextLong())

          runAll(trainingData, testData, "fs-10", nTree)
        }
      }

      if (false) {
        Seq(16, 32, 64, 128, 256, 512).foreach { nRow =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateTrainingData(nRow, 5, function = Friedman.friedmanSilverman, seed = rand.nextLong())
          val testData = TestUtils.generateTrainingData(nTest, 5, function = Friedman.friedmanSilverman, seed = rand.nextLong())

          runAll(trainingData, testData, "fs-5", nTree)
        }
      }

      if (false) {
        Seq(16, 32, 64, 128, 256, 512).foreach { nRow =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateTrainingData(nRow, 10, function = Friedman.friedmanGrosseSilverman, seed = rand.nextLong())
          val testData = TestUtils.generateTrainingData(nTest, 10, function = Friedman.friedmanGrosseSilverman, seed = rand.nextLong())

          runAll(trainingData, testData, "fgs-10", nTree)
        }
      }

      if (false) {
        Seq(16, 32, 64, 128, 256, 512).foreach { nRow =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateTrainingData(nRow, 10, function = Friedman.friedmanGrosseSilverman, seed = rand.nextLong(), shape = "normal", xoff = 0.5)
          val testData = TestUtils.generateTrainingData(nTest, 10, function = Friedman.friedmanGrosseSilverman, seed = rand.nextLong(), shape = "normal", xoff = 0.5)

          runAll(trainingData, testData, "fgs-10-normal", nTree)
        }
      }

      if (false) {
        Seq(16, 32, 64, 128, 256, 512).foreach { nRow =>
          val trainingData: Seq[(Vector[Any], Double)] = TestUtils.generateTrainingData(nRow, 32, function = linear.apply, seed = rand.nextLong())
          val testData = TestUtils.generateTrainingData(nTest, 32, function = linear.apply, seed = rand.nextLong())

          runAll(trainingData, testData, "linear-32", nTree)
        }
      }
    }
  }
}
