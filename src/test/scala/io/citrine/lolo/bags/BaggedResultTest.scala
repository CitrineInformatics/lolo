package io.citrine.lolo.bags

import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

class BaggedResultTest {

  @Test
  def testSingleMultiConsistency(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(512, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )

    val DTLearner = RegressionTreeLearner(numFeatures = 12)
    val biasLearner = RegressionTreeLearner(maxDepth = 5, leafLearner = Some(GuessTheMeanLearner()))

    Array(
      Bagger(DTLearner, numBags = 64, biasLearner = None, uncertaintyCalibration = false, useJackknife = true),
      Bagger(DTLearner, numBags = 64, biasLearner = Some(biasLearner), uncertaintyCalibration = true, useJackknife = false),
      Bagger(DTLearner, numBags = 64, biasLearner = Some(biasLearner), uncertaintyCalibration = true, useJackknife = true),
      Bagger(DTLearner, numBags = 64, biasLearner = None, uncertaintyCalibration = false, useJackknife = false)
    ).foreach { bagger =>
      testConsistency(trainingData, bagger.train(trainingData).getModel())
    }
  }

  /**
    * Confirm that a trained model provides the same uncertainty estimates when predicting a single candidate at a time
    * when compared with batch prediction.
    *
    * @param trainingData The original training data for the model
    * @param model        The trained model
    */
  private def testConsistency(trainingData: Seq[(Vector[Any], Any)], model: BaggedModel): Unit = {
    val testSubset = Random.shuffle(trainingData).take(16)
    val (singleValues, singleUncertainties) = testSubset.map { case (x, _) =>
      val res = model.transform(Seq(x))
      (res.getExpected().head.asInstanceOf[Double], res.getUncertainty().get.head.asInstanceOf[Double])
    }.unzip

    val (multiValues, multiUncertainties) = {
      val res = model.transform(testSubset.map(_._1))
      (res.getExpected().map(_.asInstanceOf[Double]), res.getUncertainty().get.map(_.asInstanceOf[Double]))
    }

    singleValues.zip(multiValues).foreach { case (x, y) => assert(Math.abs(x - y) < 1.0e-9, s"$x was not $y") }
    var idx = 0
    singleUncertainties.zip(multiUncertainties).foreach { case (x, y) =>
      assert(Math.abs(x - y) < 1.0e-9, s"$x was not $y for $idx")
      idx = idx + 1
    }
  }
}
