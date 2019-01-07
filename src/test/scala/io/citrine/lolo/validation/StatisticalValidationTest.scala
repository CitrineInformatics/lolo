package io.citrine.lolo.validation

import io.citrine.lolo.TestUtils
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.stats.functions.Linear
import org.junit.Test
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.BitmapEncoder.BitmapFormat

import scala.util.Random

class StatisticalValidationTest {

  @Test
  def testCalibration(): Unit = {
    val nFeature = 2

    val data = TestUtils.iterateTrainingData(nFeature, Linear(Seq(1.0)).apply, seed = Random.nextLong())
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.offDiagonal(nFeature).apply)
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.randomDirection(nFeature).apply)
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Friedman.friedmanSilverman)

    if (false) {
      val learner = RandomForest(numTrees = 1024)
      val metrics = StatisticalValidation.generativeValidation[Double](
        data,
        learner,
        nTrain = 32,
        nTest = 1024,
        nRound = 1,
        Map("rmse" -> RootMeanSquareError, "confidence" -> StandardConfidence, "error" -> StandardError))
      metrics.foreach { case (name, (mean, sigma)) =>
        println(f"$name%15s: $mean%6.4f +/- $sigma%6.4f")
      }
    } else {
      Seq(256).foreach { nTrain =>
        Seq(256, 512, 1024).foreach { nTree =>
          val learner = RandomForest(numTrees = nTree)
          val chart = StatisticalValidation.generativeHistogram(
            data,
            learner,
            nTrain = nTrain,
            nTest = 512,
            nRound = 64
          )
          BitmapEncoder.saveBitmap(chart, s"./stderr_${nTrain}_${nTree}", BitmapFormat.PNG)
        }
      }
    }
  }

}
