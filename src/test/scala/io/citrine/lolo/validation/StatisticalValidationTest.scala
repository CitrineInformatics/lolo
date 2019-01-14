package io.citrine.lolo.validation

import io.citrine.lolo.TestUtils
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.stats.functions.{Friedman, Linear}
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
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Friedman.friedmanSilverman)

    if (true) {
      val nTrain = 16

      val chart = Metric.scanMetrics(
        "Number of Trees",
        Seq(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        Map("rmse" -> RootMeanSquareError, "confidence" -> StandardConfidence, "error" -> StandardError),
        logScale = true
      ){ nTrees: Double =>
        val learner = RandomForest(numTrees = nTrees.toInt)
        StatisticalValidation.generativeValidation[Double](
        data,
        learner,
        nTrain = nTrain,
        nTest = nTrain,
        nRound = 64)
      }
      BitmapEncoder.saveBitmap(chart, s"./metric_scan_${nTrain}", BitmapFormat.PNG)
    } else {
//      Seq(16, 32, 64, 128, 256, 512, 1024, 2048).foreach { nTrain =>
//        Seq(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048).foreach { nTree =>
      Seq(64).foreach { nTrain =>
        Seq(64).foreach { nTree =>
          val learner = RandomForest(numTrees = nTree)
          val pva = StatisticalValidation.generativeValidation(
            data,
            learner,
            nTrain = nTrain,
            nTest = 512,
            nRound = 64
          )
          val chart = StandardResidualHistogram().visualize(pva)
          BitmapEncoder.saveBitmap(chart, s"./stderr_sweep_${nTrain}_${nTree}", BitmapFormat.PNG)
        }
      }
    }
  }

}
