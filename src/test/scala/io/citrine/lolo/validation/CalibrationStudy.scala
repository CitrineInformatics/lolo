package io.citrine.lolo.validation

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.stats.functions.{Friedman, Linear}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.{Learner, PredictionResult, TestUtils}
import org.apache.commons.math3.distribution.CauchyDistribution
import org.knowm.xchart.BitmapEncoder.BitmapFormat
import org.knowm.xchart.{BitmapEncoder, CategoryChart, CategoryChartBuilder}

import scala.collection.JavaConverters._
import scala.util.Random

object CalibrationStudy {

  def main(args: Array[String]): Unit = {

    val generateFollowUps = false
    if (generateFollowUps) {
      generateSweepAtFixedTreeCount(calibrated = false, func = Friedman.friedmanGrosseSilverman, funcName = "fgs")
      generateSweepAtFixedTreeCount(calibrated = true, func = Friedman.friedmanGrosseSilverman, funcName = "fgs")
      generateSweepForHCEP(calibrated = false)
      generateSweepForHCEP(calibrated = true)
      Seq(16, 32, 64, 128, 256, 512).foreach{nTrain =>
        generatePvaAndHistogramHCEP(nTrain, nTree = 128)
      }
      generatePvaAndHistogram(Friedman.friedmanSilverman, "fs-trunc-bias", nTrain = 1024, nTree = 64, range = 2, ignoreDims = 1)
    }

    val generateFiguresForPaper = false
    if (generateFiguresForPaper) {
      generatePvaAndHistogram(Linear(Seq(1.0)).apply, "lin", nTrain = 64, nTree = 64)
      println("Metrics for FS with 64 points and 64 trees:")
      println(generatePvaAndHistogram(Friedman.friedmanSilverman, "fs", nTrain = 64, nTree = 64))
      generatePvaAndHistogram(Friedman.friedmanSilverman, "fs", nTrain = 16, nTree = 64, range = 10)
      println("Metrics for FS with 64 points and 16 trees:")
      println(generatePvaAndHistogram(Friedman.friedmanSilverman, "fs", nTrain = 64, nTree = 16))
      generateSweepAtFixedRatio()
      generateSweepAtFixedTrainingSize(calibrated = false)
      generateSweepAtFixedTrainingSize(calibrated = true)
    }

  }

  def generateSweepAtFixedRatio(
                                 ratios: Seq[Int] = Seq(1, 2, 4)
                               ): Unit = {
    val nFeature = 8
    val data = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman, seed = Random.nextLong())

    ratios.foreach { ratio =>
      val chart = Merit.plotMeritScan(
        "Number of training rows",
        Seq(16, 32, 64, 128, 256, 512),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "standard error / 4" -> StandardError(0.25), "sigmaCorr" -> UncertaintyCorrelation),
        logScale = true,
        yMin = Some(0.0),
        yMax = Some(1.0)
      ) { nTrain: Double =>
        val nTree = ratio * nTrain
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature
          ),
          numBags = nTree.toInt,
          useJackknife = true,
          uncertaintyCalibration = false
        )
        StatisticalValidation.generativeValidation[Double](
          data,
          learner,
          nTrain = nTrain.toInt,
          nTest = 256,
          nRound = 32)
      }
      BitmapEncoder.saveBitmap(chart, s"./scan-ratio.${ratio}", BitmapFormat.PNG)
    }
  }

  def generateSweepAtFixedTrainingSize(
                                        sizes: Seq[Int] = Seq(16, 32, 64, 128, 256),
                                        calibrated: Boolean = false
                                      ): Unit = {
    val nFeature = 8
    val data = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman, seed = Random.nextLong())
    sizes.foreach { nTrain =>
      val chart = Merit.plotMeritScan(
        "Number of trees",
        Seq(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "standard error / 4" -> StandardError(0.25), "sigmaCorr" -> UncertaintyCorrelation),
        logScale = true,
        yMin = Some(0.0),
        yMax = Some(1.0)
      ) { nTree: Double =>
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature
          ),
          numBags = nTree.toInt,
          useJackknife = true,
          uncertaintyCalibration = calibrated
        )
        StatisticalValidation.generativeValidation[Double](
          data,
          learner,
          nTrain = nTrain.toInt,
          nTest = 256,
          nRound = 32)
      }
      val fname = if (calibrated) {
        s"./scan-nTrain.${nTrain}-nTree-cal"
      } else {
        s"./scan-nTrain.${nTrain}-nTree"
      }
      BitmapEncoder.saveBitmap(chart, fname, BitmapFormat.PNG)
    }
  }

  def generateSweepAtFixedTreeCount(
                                     sizes: Seq[Int] = Seq(64, 128, 256),
                                     calibrated: Boolean = false,
                                     func: Seq[Double] => Double = Friedman.friedmanSilverman,
                                     funcName: String = "fs"
                                   ): Unit = {
    val nFeature = 8
    val data = TestUtils.iterateTrainingData(nFeature, func, seed = Random.nextLong())
    sizes.foreach { nTree =>
      val chart = Merit.plotMeritScan(
        "Number of training points",
        Seq(16, 32, 64, 128, 256, 512, 1024),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "standard error / 4" -> StandardError(0.25), "sigmaCorr" -> UncertaintyCorrelation),
        logScale = true,
        yMin = Some(0.0),
        yMax = Some(1.0)
      ) { nTrain: Double =>
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature
          ),
          numBags = nTree.toInt,
          useJackknife = true,
          uncertaintyCalibration = calibrated
        )
        StatisticalValidation.generativeValidation[Double](
          data,
          learner,
          nTrain = nTrain.toInt,
          nTest = 256,
          nRound = 32)
      }
      val fname = if (calibrated) {
        s"./scan-${funcName}-nTrain-nTree.${nTree}-cal"
      } else {
        s"./scan-${funcName}-nTrain-nTree.${nTree}"
      }
      BitmapEncoder.saveBitmap(chart, fname, BitmapFormat.PNG)
    }
  }


  def generatePvaAndHistogram(
                               func: Seq[Double] => Double = Friedman.friedmanSilverman,
                               funcName: String = "fs",
                               nTrain: Int = 64,
                               nTree: Int = 16,
                               nFeature: Int = 8,
                               range: Double = 4.0,
                               ignoreDims: Int = 0
                             ): Map[String, (Double, Double)] = {

    val data = TestUtils.iterateTrainingData(nFeature, func, seed = Random.nextLong())
      .map{case (x, y) => (x.drop(ignoreDims), y)}
    val learner = Bagger(
      RegressionTreeLearner(
        numFeatures = nFeature / 3
      ),
      numBags = nTree,
      useJackknife = true,
      uncertaintyCalibration = false,
      biasLearner = Some(RegressionTreeLearner(maxDepth = (Math.log(nTrain) / Math.log(2) / 2).toInt))
    )

    val fullStream = StatisticalValidation.generativeValidation[Double](
      data,
      learner,
      nTrain,
      32,
      512
    ).toIterable

    val chart = generativeHistogram(fullStream, learner,
      plotNormal = true,
      plotCauchy = true,
      calibrate = true,
      range = range
    )
    BitmapEncoder.saveBitmap(chart, s"./stdres_${funcName}-nTrain.${nTrain}-nTree.${nTree}", BitmapFormat.PNG)

    val dataStream = StatisticalValidation.generativeValidation[Double](
      data,
      learner,
      nTrain,
      32, 2
    ).toIterable
    val pva = PredictedVsActual().visualize(dataStream)
    BitmapEncoder.saveBitmap(pva, s"./pva_${funcName}-nTrain.${nTrain}-nTree.${nTree}", BitmapFormat.PNG)
    Merit.estimateMerits(
      dataStream.iterator,
      Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "error / 4" -> StandardError(0.25), "sigmaCorr" -> UncertaintyCorrelation)
    )
  }

  def generativeHistogram(
                           source: Iterable[(PredictionResult[Double], Seq[Double])],
                           learner: Learner,
                           plotCauchy: Boolean = false,
                           plotNormal: Boolean = false,
                           calibrate: Boolean = true,
                           range: Double = 4.0
                         ): CategoryChart = {
    val data: Seq[(Double, Double, Double)] = source.flatMap{ case(predictions, actual) =>
      (predictions.getExpected(), predictions.getUncertainty().get.asInstanceOf[Seq[Double]], actual).zipped.toSeq
    }.toSeq

    val standardErrors = data.map { case (predicted, sigma, actual) => (predicted - actual) / sigma }

    val span = 2 * range // Math.ceil(standardErrors.map(Math.abs).max).toInt.toDouble
    val nBins = 128
    val bins = (-nBins / 2 until nBins / 2).map { idx =>
      (span * idx.toDouble / nBins, span * (idx.toDouble + 1) / nBins)
    }

    val counts = bins.map { case (min, max) =>
      ((min + max) / 2.0, standardErrors.count(x => x >= min && x < max) / (standardErrors.size * (max - min)))
    }

    val normalVar = if (calibrate) {
      Math.pow(standardErrors.map(Math.abs(_)).sorted.drop((0.68 * standardErrors.size).toInt).head, 2)
    } else {
      1.0
    }

    val chart: CategoryChart = new CategoryChartBuilder().build()
    chart.addSeries("data", counts.map(_._1).toArray, counts.map(_._2.toDouble).toArray)

    val normalSeries = counts.map(_._1).map(x => Math.exp(-x * x / (2 * normalVar)) / Math.sqrt(2 * Math.PI * normalVar))
    if (plotNormal) {
      chart.addSeries(f"Normal(0, ${Math.sqrt(normalVar)}%6.3f)", counts.map(_._1).toArray, normalSeries.toArray)
    }

    val halfWidth = standardErrors.map(Math.abs(_)).sorted.drop((0.5 * standardErrors.size).toInt).head
    val gamma: Double = if (calibrate) {
      halfWidth
    } else {
      1.0
    }
    val cauchy1 = new CauchyDistribution(0.0, gamma)
    val cauchySeries = counts.map(_._1).map(x => cauchy1.density(x))
    if (plotCauchy) {
      chart.addSeries(f"Cauchy(${gamma}%6.3f)", counts.map(_._1).toArray, cauchySeries.toArray)
    }

    chart.setTitle("(predicted - actual) / (predicted uncertainty)")
    chart.setXAxisLabelOverrideMap(Map[java.lang.Double, AnyRef]().asJava)
    chart.setYAxisTitle("probability density")
    chart
  }

  def generateSweepForHCEP(
                                sizes: Seq[Int] = Seq(64, 128, 256),
                                calibrated: Boolean = false
                          ): Unit = {
    val csv = TestUtils.readCsv("hcep.csv")
    val trainingData = csv.tail.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
    val nFeature = 8
    sizes.foreach { nTree =>
      val chart = Merit.plotMeritScan(
        "Number of training points",
        Seq(16, 32, 64, 128, 256, 512, 1024),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "standard error / 4" -> StandardError(0.25), "sigmaCorr" -> UncertaintyCorrelation),
        logScale = true,
        yMin = Some(0.0),
        yMax = Some(1.0)
      ) { nTrain: Double =>
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature
          ),
          numBags = nTree.toInt,
          useJackknife = true,
          uncertaintyCalibration = calibrated
        )
        StatisticalValidation.generativeValidation[Double](
          trainingData,
          learner,
          nTrain = nTrain.toInt,
          nTest = 256,
          nRound = 32)
      }
      val fname = if (calibrated) {
        s"./scan-hcep-nTrain-nTree.${nTree}-cal"
      } else {
        s"./scan-hcep-nTrain-nTree.${nTree}"
      }
      BitmapEncoder.saveBitmap(chart, fname, BitmapFormat.PNG)
    }
  }

  def generatePvaAndHistogramHCEP(
                               nTrain: Int = 64,
                               nTree: Int = 16,
                               nFeature: Int = 8,
                               range: Double = 4.0
                             ): Map[String, (Double, Double)] = {

    val csv = TestUtils.readCsv("hcep.csv")
    val trainingData = csv.tail.map(vec => (vec.init, vec.last.asInstanceOf[Double]))

    val learner = Bagger(
      RegressionTreeLearner(
        numFeatures = nFeature / 3
      ),
      numBags = nTree,
      useJackknife = true,
      uncertaintyCalibration = false
    )

    val fullStream = StatisticalValidation.generativeValidation[Double](
      trainingData,
      learner,
      nTrain,
      32, 512
    ).toIterable

    val chart = generativeHistogram(fullStream, learner,
      plotNormal = true,
      plotCauchy = true,
      calibrate = true,
      range = range
    )
    BitmapEncoder.saveBitmap(chart, s"./stdres_hcep-nTrain.${nTrain}-nTree.${nTree}", BitmapFormat.PNG)

    val dataStream = StatisticalValidation.generativeValidation[Double](
      trainingData,
      learner,
      nTrain,
      32, 2
    ).toIterable
    val pva = PredictedVsActual().visualize(dataStream)
    BitmapEncoder.saveBitmap(pva, s"./pva_hcep-nTrain.${nTrain}-nTree.${nTree}", BitmapFormat.PNG)
    Merit.estimateMerits(
      dataStream.iterator,
      Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "error / 4" -> StandardError(0.25), "sigmaCorr" -> UncertaintyCorrelation)
    )
  }

}
