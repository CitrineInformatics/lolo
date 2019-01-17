package io.citrine.lolo.validation

import io.citrine.lolo.TestUtils
import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.{Friedman, Linear}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.BitmapEncoder.BitmapFormat

import scala.util.Random

class StatisticalValidationTest {

  @Test
  def testCalibration(): Unit = {
    val nFeature = 2

    // val data = TestUtils.iterateTrainingData(nFeature, Linear.offDiagonal(nFeature).apply, seed = Random.nextLong())
    val data = TestUtils.iterateTrainingData(nFeature, Linear(Seq(1.0)).apply, seed = Random.nextLong())
    // val data = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman, seed = Random.nextLong())
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.offDiagonal(nFeature).apply)
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.randomDirection(nFeature).apply)
    // val data = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman)

    if (true) {
      val nTrain = 32

      val chart = Metric.scanMetrics(
        "Number of Trees",
        Seq(16, 32, 64, 128, 256),
        // Seq(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence, "error" -> StandardError, "sigmaCorr" -> UncertaintyCorrelation),
        logScale = true
      ){ nTrees: Double =>
        val nTrain = nTrees.toInt
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = 8
          ),
          numBags = nTrees.toInt,
          useJackknife = true,
          biasLearner = Some(RegressionTreeLearner(maxDepth = 2))
        )
        StatisticalValidation.generativeValidation[Double](
        data,
        learner,
        nTrain = nTrain,
        nTest = nTrain,
        nRound = 128)
      }
      BitmapEncoder.saveBitmap(chart, s"./metric_scan2_${nTrain}", BitmapFormat.PNG)
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
          val errorChart = ErrorvsUncertainty().visualize(pva)
          BitmapEncoder.saveBitmap(errorChart, s"./errcor_sweep_${nTrain}_${nTree}", BitmapFormat.PNG)
        }
      }
    }
  }

}
