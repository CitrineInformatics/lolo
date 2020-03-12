package io.citrine.lolo.bags

import io.citrine.lolo.{RegressionResult, TestUtils}
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

  @Test
  def testBaggedSingleResultGetUncertainty(): Unit = {
    val noiseLevel = 100.0
    val rng = new Random(237485L)
    Seq(RegressionTreeLearner(),GuessTheMeanLearner()).foreach{ baseLearner =>
      Seq(30,100,301).foreach { nRows =>
        val trainingDataTmp = TestUtils.generateTrainingData(nRows, 1, noise = 0.0, function = _ => 0.0, seed = rng.nextLong())
        val trainingData = (trainingDataTmp).map { x => (x._1, x._2 + noiseLevel * rng.nextDouble()) }
        val baggedLearner = Bagger(baseLearner, numBags = 2 * nRows, uncertaintyCalibration = true)
        val RFMeta = baggedLearner.train(trainingData)
        val RF = RFMeta.getModel()
        val results = RF.transform(trainingData.take(1).map(_._1))

        val sigmaObs: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
//        sigmaMean.zip(results.asInstanceOf[RegressionResult].getStdDevMean().get).foreach{ case (a,b) =>
//          assert(a == b, "Expected getUncertainty(observational=false)=getStdDevMean()")
//        }
//        val sigmaMean: Seq[Double] = results.getUncertainty(observational = false).get.asInstanceOf[Seq[Double]]
        sigmaObs.zip(results.asInstanceOf[RegressionResult].getStdDevObs().get).foreach{ case (a,b) =>
          assert(a == b, "Expected getUncertainty()=getStdDevObs()")
        }

//        sigmaObs.zip(sigmaMean).foreach { case (sObs, sMean) => assert(sObs > sMean, "Uncertainty should be greater when observational = true.") }

        // We have strong theoretical guarantees on the behavior of GuessTheMeanLearner, so let's exercise them.
        // TODO(grobinson): implement similar test for RegressionTreeLearner.
        if (baseLearner.isInstanceOf[GuessTheMeanLearner]) {
          // NOTE: these bounds reflect a ~3x systematic variance under-estimation in this particular test setting.
          val rtolLower = 5.0  // Future recalibration should decrease this number.
          val rtolUpper = 1.0  // Future recalibration should increase this number.
          sigmaObs.foreach { s =>
            assert(rtolLower * s > noiseLevel, "Observational StdDev getUncertainty() is too small.")
            assert(s < rtolUpper * noiseLevel, "Observational StdDev getUncertainty() is too large.")
          }
//          sigmaMean.foreach { s =>
//            assert(rtolLower * s > noiseLevel / Math.sqrt(nRows), "Mean StdDev getUncertainty(observational=false) is too small.")
//            assert(s < rtolUpper * noiseLevel / Math.sqrt(nRows), "Mean StdDev getUncertainty(observational=false) is too large.")
//          }
        }
      }
    }
  }

  /**
    * Confirm that a trained model provides the same uncertainty estimates when predicting a single candidate at a time
    * when compared with batch prediction.
    *
    * @param trainingData The original training data for the model
    * @param model        The trained model
    */
  private def testConsistency(trainingData: Seq[(Vector[Any], Any)], model: BaggedModel[Any]): Unit = {
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
