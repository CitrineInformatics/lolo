package io.citrine.lolo.validation

import io.citrine.lolo.{Learner, PredictionResult, TestUtils}
import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.stats.functions.{Friedman}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.apache.commons.math3.distribution.{CauchyDistribution, MultivariateNormalDistribution}
import org.junit.Test
import org.knowm.xchart.{BitmapEncoder, CategoryChart, CategoryChartBuilder}
import org.knowm.xchart.BitmapEncoder.BitmapFormat

import scala.util.{Random}
import scala.collection.JavaConverters._

class StatisticalValidationTest {

  /**
    * Check that the generative validation is approximately consistent with k-fold cross-validation
    *
    * This test is not supposed to be statistically rigorous; just specific enough to detect bugs in the implementation
    */
  @Test
  def testCompareToKFolds(): Unit = {
    val learner = RandomForest()
    val dataSet = TestUtils.generateTrainingData(128, 8, Friedman.friedmanSilverman)
    val dataGenerator = TestUtils.iterateTrainingData(8, Friedman.friedmanSilverman)

    val metrics = Map("rmse" -> RootMeanSquareError)
    val (rmseFromCV, uncertaintyFromCV) = CrossValidation.kFoldCrossvalidation(dataSet, learner, metrics, k = 4, nTrial = 4)("rmse")

    val (rmseFromStats, uncertaintyFromStats) = Metric.estimateMetrics(
      StatisticalValidation.generativeValidation(dataGenerator, learner, 96, 32, 16),
      metrics
    )("rmse")

    // These assertions have a false negative rate of 1/1000
    assert(
      uncertaintyFromCV + uncertaintyFromStats < (rmseFromCV + rmseFromStats) / 4.0,
      s"The uncertainty is more than half the metric value; that's too uncertain to test"
    )
    assert(
      Math.abs(rmseFromCV - rmseFromStats) < 2.0 * (uncertaintyFromCV + uncertaintyFromStats),
      s"The CV and statistical validation methods disagree more than 2 sigma-sum"
    )
  }

  @Test
  def testCalibration(): Unit = {
    val nFeature = 8

    // val data = TestUtils.iterateTrainingData(nFeature, Linear.offDiagonal(nFeature).apply, seed = Random.nextLong())
    // val data = TestUtils.iterateTrainingData(nFeature, Linear(Seq(1.0)).apply, seed = Random.nextLong())
    val data = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman, seed = Random.nextLong())
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.offDiagonal(nFeature).apply)
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.randomDirection(nFeature).apply)
    // val data = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman)

    if (true) {
      val nTrain = 128
      val nTree = 256

      val chart = Metric.plotMetricScan(
        "Number of Training Samples",
        Seq(16, 32, 64, 128, 256, 512),
        // Seq(2, 4, 8, 16, 32, 64, 128, 256, 512),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "error" -> StandardError, "sigmaCorr" -> UncertaintyCorrelation),
        logScale = true
      ){ nTrain: Double =>
        // val nTrain = nTrees.toInt
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature
          ),
          numBags = nTree,
          useJackknife = true,
          biasLearner = Some(RegressionTreeLearner(maxDepth = 2))
        )
        StatisticalValidation.generativeValidation[Double](
        data,
        learner,
        nTrain = nTrain.toInt,
        nTest = 256,
        nRound = 32)
      }
      BitmapEncoder.saveBitmap(chart, s"./metric_scan3_${nTrain}", BitmapFormat.PNG)
    } else {
//      Seq(16, 32, 64, 128, 256, 512, 1024, 2048).foreach { nTrain =>
//        Seq(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048).foreach { nTree =>
      Seq(32).foreach { nTrain =>
        Seq(4096).foreach { nTree =>
          val learner = RandomForest(numTrees = nTree)
          val pva = StatisticalValidation.generativeValidation(
            data,
            learner,
            nTrain = nTrain,
            nTest = 16,
            nRound = 512
          )
          println(s"CorrelationCoeff is ${UncertaintyCorrelation.estimate(pva)}")
          val chart = StandardResidualHistogram().visualize(pva)
          BitmapEncoder.saveBitmap(chart, s"./stderr_sweep_${nTrain}_${nTree}", BitmapFormat.PNG)
          val pvaChart = PredictedVsActual().visualize(pva)
          BitmapEncoder.saveBitmap(pvaChart, s"./pva_sweep_${nTrain}_${nTree}", BitmapFormat.PNG)
          val errorChart = ErrorVsUncertainty().visualize(pva)
          BitmapEncoder.saveBitmap(errorChart, s"./errcor_sweep_${nTrain}_${nTree}", BitmapFormat.PNG)
        }
      }
    }
  }

  private def computeVariances(error: Seq[Double], sigma: Seq[Double]): (Double, Double, Double) = {
    require(sigma.forall(_ >= 0.0), sigma.min)
    val xRec: Seq[Double] = error.map(Math.abs)
    val xBar = xRec.sum / error.size
    val yBar = sigma.sum / sigma.size
    val varX = xRec.map(x => Math.pow(x - xBar, 2.0)).sum / error.size + xBar * xBar
    val varY = sigma.map(y => Math.pow(y - yBar, 2.0)).sum / sigma.size + yBar * yBar
    val varXY = xRec.zip(sigma).map{case (x, y) => (x - xBar) * (y - yBar)}.sum / error.size + yBar * xBar

    (varX, varY, varXY)
  }

  def generativeHistogram(
                               source: Iterator[(Vector[Any], Double)],
                               learner: Learner,
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int
                             ): CategoryChart = {
    val data: Seq[(Double, Double, Double)] = (0 until nRound).flatMap { _ =>
      val trainingData: Seq[(Vector[Any], Double)] = source.take(nTrain).toSeq
      val model = learner.train(trainingData).getModel()
      val testData: Seq[(Vector[Any], Double)] = source.take(nTest).toSeq
      val predictions: PredictionResult[Double] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[Double]]
      (predictions.getExpected(), predictions.getUncertainty().get.asInstanceOf[Seq[Double]], testData.map(_._2)).zipped.toSeq
    }

    val absoluteErrors = data.map { case (predicted, _, actual) =>
      Math.abs(predicted - actual)
    }

    val (varError, varSigma, sigmaCorr) = computeVariances(absoluteErrors, data.map(_._2))

    println(s"The covariance of sigma and absolute error is ${sigmaCorr / Math.sqrt(varSigma * varError)}")

    val standardErrors = data.map { case (predicted, sigma, actual) => (predicted - actual) / sigma }

    val range = 8.0 // Math.ceil(standardErrors.map(Math.abs).max).toInt.toDouble
    val nBins = 128
    val bins = (-nBins / 2 until nBins / 2).map { idx =>
      (range * idx.toDouble / nBins, range * (idx.toDouble + 1) / nBins)
    }

    val counts = bins.map { case (min, max) =>
      ((min + max) / 2.0, standardErrors.count(x => x >= min && x < max) / (standardErrors.size * (max - min)))
    }

    val crossTerm = 0.0 * Math.sqrt(varError * varSigma) + 1.0 * sigmaCorr
    val meanSigma = data.map(_._2).sum / data.size
    val multiDist = new MultivariateNormalDistribution(Array(0.0, 0.0), Array(Array(varError, crossTerm), Array(crossTerm, varSigma)))
    val fakeRaw = if (true) {
      Seq.fill(data.size) {
        val point = multiDist.sample()
        (point(0) * Random.nextGaussian(), Math.abs(point(1)))
      }
    } else {
      Seq.fill(data.size) {
        val sigma = Math.abs(Random.nextGaussian() * Math.sqrt(varSigma))
        val sigma2 = Math.abs(Random.nextGaussian() * Math.sqrt(varSigma))
        val error = Random.nextGaussian() * Math.sqrt(varError)
        (error, sigma)
      }
    }

    // println(varError, varSigma, sigmaCorr)
    val fakeCorr = computeVariances(fakeRaw.map(_._1), fakeRaw.map(_._2))._3
    println(f"Empirical vs model cross-correlation: ${sigmaCorr}%7.4f vs ${fakeCorr}%7.4f")

    val fakeData: Seq[Double] = if (true) {
      fakeRaw.map{case (x, y) =>
        x / y
      }
    } else {
      data.map(_._2 / Math.sqrt(varSigma))
    }

    val fakeCounts = bins.map{case (min, max) =>
      ((min + max) / 2.0, fakeData.count(x => x >= min && x < max) / (standardErrors.size * (max - min)))
    }

    val normalVar = Math.pow(standardErrors.map(Math.abs(_)).sorted.drop((0.68 * standardErrors.size).toInt).head, 2)
    val halfWidth = standardErrors.map(Math.abs(_)).sorted.drop((0.5 * standardErrors.size).toInt).head

    val chart: CategoryChart = new CategoryChartBuilder().build()
    chart.addSeries("data", counts.map(_._1).toArray, counts.map(_._2.toDouble).toArray)
    val normalSeries = counts.map(_._1).map(x => Math.exp(-x*x / (2 * normalVar) )/Math.sqrt(2 * Math.PI * normalVar))
    // chart.addSeries(f"sigma=${Math.sqrt(normalVar)}%6.3f", counts.map(_._1).toArray, normalSeries.toArray)
    // val tdist = new TDistribution(1)
    //c hart.addSeries("t=1", counts.map(_._1).toArray, counts.map(_._1).map(x => tdist.density(x)).toArray)
    val gamma = halfWidth //  Math.sqrt(standardErrors.map(Math.pow(_, 2.0)).sum / standardErrors.size)
    val cauchy1 = new CauchyDistribution(0.0, gamma)
    val cauchySeries = counts.map(_._1).map(x => cauchy1.density(x))
    // chart.addSeries(f"gamma=${gamma}%6.3f", counts.map(_._1).toArray, cauchySeries.toArray)
    // chart.addSeries("synth", fakeCounts.map(_._1).toArray, fakeCounts.map(_._2.toDouble).toArray)
    val correlationCoefficient = sigmaCorr / Math.sqrt(varSigma * varError)
    val mixtureSeries = normalSeries.zip(cauchySeries).map{case (n, c) =>
      n * correlationCoefficient + (1 - correlationCoefficient) * c
    }
    chart.addSeries("mixture", fakeCounts.map(_._1).toArray, mixtureSeries.toArray)
    val gamma3 = Math.sqrt(varError / varSigma)
    // val cauchy3 = new CauchyDistribution(0.0, gamma3)
    // chart.addSeries(s"gamma=${gamma3}", counts.map(_._1).toArray, counts.map(_._1).map(x => cauchy3.density(x)).toArray)

    chart.setTitle("(predicted - actual) / (predicted uncertainty)")
    chart.setXAxisLabelOverrideMap(Map[java.lang.Double, AnyRef]().asJava)
    chart.setYAxisTitle("probability density")
    chart
  }

}
