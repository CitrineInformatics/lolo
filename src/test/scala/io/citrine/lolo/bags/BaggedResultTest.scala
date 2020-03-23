package io.citrine.lolo.bags

import breeze.stats.distributions.Beta
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

  /**
    * Test that uncertainty estimates are within reasonable bounds.
    */
  @Test
  def testBaggedSingleResultGetUncertainty(): Unit = {
    val noiseLevel = 100.0
    val rng = new Random(237485L)

    Seq(RegressionTreeLearner(), GuessTheMeanLearner()).foreach { baseLearner =>
      // These are in Seqs as a convenience for repurposing this test as a diagnostic tool.
      Seq(128).foreach { nRows =>
        Seq(16).foreach { nCols =>
          Seq(2).map { n => n * nRows }.foreach { nBags =>
            // Used for error output.
            val configDescription = s"learner=${baseLearner.getClass().toString()}\tnRows=$nRows\tnCols=$nCols\tnumBags=$nBags"

            val sigmaObsAndSigmaMean: Seq[(Double, Double)] = (1 to 20).flatMap { _ =>
              val trainingDataTmp = TestUtils.generateTrainingData(nRows, nCols, noise = 0.0, function = _ => 0.0, seed = rng.nextLong())
              val trainingData = (trainingDataTmp).map { x => (x._1, x._2 + noiseLevel * rng.nextDouble()) }
              val baggedLearner = Bagger(baseLearner, numBags = nBags, uncertaintyCalibration = true)
              val RFMeta = baggedLearner.train(trainingData)
              val RF = RFMeta.getModel()
              val results = RF.transform(trainingData.take(4).map(_._1))

              val sigmaMean: Seq[Double] = results.getUncertainty(observational = false).get.asInstanceOf[Seq[Double]]
              sigmaMean.zip(results.asInstanceOf[RegressionResult].getStdDevMean().get).foreach { case (a, b) =>
                assert(a == b, s"Expected getUncertainty(observational=false)=getStdDevMean() for $configDescription")
              }

              val sigmaObs: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
              sigmaObs.zip(results.asInstanceOf[RegressionResult].getStdDevObs().get).foreach { case (a, b) =>
                assert(a == b, s"Expected getUncertainty()=getStdDevObs() for $configDescription")
              }

              // We have strong theoretical guarantees on the behavior of GuessTheMeanLearner, so let's exercise them.
              // NOTE: these bounds reflect a ~3x systematic variance under-estimation in this particular test setting.
              {
                val rtolLower = baseLearner match {
                  case _: GuessTheMeanLearner => 3.5
                  case _: Any => 10.0
                }
                val rtolUpper = baseLearner match {
                  case _: GuessTheMeanLearner => 1.0
                  case _: Any => 1.0
                }
                sigmaObs.foreach { s =>
                  assert(rtolLower * s > noiseLevel, s"Observational StdDev getUncertainty() is too small for $configDescription")
                  assert(s < rtolUpper * noiseLevel, s"Observational StdDev getUncertainty() is too large for $configDescription")
                }
              }
              {
                val rtolLower = baseLearner match {
                  case _: GuessTheMeanLearner => 5.0
                  case _: Any => 1e3
                }
                val rtolUpper = baseLearner match {
                  case _: GuessTheMeanLearner => 1.0
                  case _: Any => 10.0
                }
                sigmaMean.foreach { s =>
                  assert(rtolLower * s > (noiseLevel / Math.sqrt(nRows)), s"Mean StdDev getUncertainty(observational=false)=$s is too small for $configDescription.")
                  assert(s < (rtolUpper * noiseLevel / Math.sqrt(nRows)), s"Mean StdDev getUncertainty(observational=false)=$s is too large for $configDescription")
                }
              }

              // Uncomment for diagnostic output.
              // sigmaObs.zip(sigmaMean).foreach { case (sObs, sMean) =>
              //   println(s"$configDescription\tsObs=$sObs\tsMean=$sMean")
              // }

              sigmaObs.zip(sigmaMean)
            }

            val countSigmaObsGreater = sigmaObsAndSigmaMean.count { case (sObs, sMean) => sObs > sMean }.toDouble
            // Posterior beta distribution, with Jeffreys prior, over rate at which sObs > sMean.
            val d = new Beta(countSigmaObsGreater + 0.5, sigmaObsAndSigmaMean.length - countSigmaObsGreater + 0.5)
            val minRateSigmaObsGreater = 0.9
            val level = 1e-4
            val probSigmaObsLess = d.cdf(minRateSigmaObsGreater)
            assert(probSigmaObsLess < level, s"Uncertainty should be greater when observational = true for $configDescription")
          }
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
