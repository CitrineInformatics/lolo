package io.citrine.lolo.validation

import io.citrine.lolo.TestUtils
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.stats.functions.{Friedman, Linear}
import org.junit.Test

import scala.util.Random

class CrossValidationTest {

  @Test
  def testCompareToOutOfBag(): Unit = {
    val learner = RandomForest()
    val data = TestUtils.generateTrainingData(128, 8, Friedman.friedmanSilverman)

    val metric = RootMeanSquareError
    val (rmseFromCV, uncertainty) = CrossValidation.kFoldCrossvalidation(data, learner, Map("rmse" -> metric), k = 3)("rmse")

    val trainingResult = learner.train(data)
    val rmseFromPVA = Math.sqrt(
      trainingResult.getPredictedVsActual().get.map {
        case (_, p: Double, a: Double) => Math.pow(p - a, 2.0)
      }.sum / trainingResult.getPredictedVsActual().get.size
    )

    // These have a false negative rate less than 1/100 at the time of original authorship
    assert(rmseFromPVA < rmseFromCV + uncertainty)
    assert((rmseFromPVA - rmseFromCV) / rmseFromPVA < 0.2)
  }

  @Test
  def testCalibration(): Unit = {
    val nSample = 128
    val nRow = 16
    val nTree = 4096
    val nFeature = 2
    val learner = RandomForest(numTrees = nTree)

    val data = TestUtils.generateTrainingData(nRow, nFeature, Linear(Seq(1.0)).apply, seed = Random.nextLong())
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.offDiagonal(nFeature).apply)
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Linear.randomDirection(nFeature).apply)
    // val data = TestUtils.generateTrainingData(nRow, nFeature, Friedman.friedmanSilverman)

    val metrics = CrossValidation.kFoldCrossvalidation(data, learner, Map("rmse" -> RootMeanSquareError, "confidence" -> StandardConfidence, "error" -> StandardError), k = nRow, nTrial = nSample / nRow)
    metrics.foreach{case (name, (mean, sigma)) =>
        println(f"$name%15s: $mean%5.3f +/- $sigma%5.3f")
    }
  }

}
