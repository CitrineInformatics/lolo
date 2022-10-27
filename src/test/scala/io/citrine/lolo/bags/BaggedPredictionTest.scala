package io.citrine.lolo.bags

import breeze.stats.distributions.{Beta, RandBasis}
import io.citrine.lolo.api.{RegressionResult, TrainingRow}
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

class BaggedPredictionTest extends SeedRandomMixIn {

  @Test
  def testSingleMultiConsistency(): Unit = {
    val trainingData = DataGenerator
      .generate(512, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    val DTLearner = RegressionTreeLearner(numFeatures = 12)
    val biasLearner = RegressionTreeLearner(maxDepth = 5, leafLearner = Some(GuessTheMeanLearner()))

    Array(
      RegressionBagger(
        DTLearner,
        numBags = 64,
        biasLearner = None,
        uncertaintyCalibration = false,
        useJackknife = true
      ),
      RegressionBagger(
        DTLearner,
        numBags = 64,
        biasLearner = Some(biasLearner),
        uncertaintyCalibration = true,
        useJackknife = false
      ),
      RegressionBagger(
        DTLearner,
        numBags = 64,
        biasLearner = Some(biasLearner),
        uncertaintyCalibration = true,
        useJackknife = true
      ),
      RegressionBagger(
        DTLearner,
        numBags = 64,
        biasLearner = None,
        uncertaintyCalibration = false,
        useJackknife = false
      )
    ).foreach { bagger =>
      testConsistency(trainingData, bagger.train(trainingData, rng = rng).model)
    }
  }

  /**
    * Test that uncertainty estimates are within reasonable bounds.
    */
  @Test
  def testBaggedSingleResultGetUncertainty(): Unit = {
    val noiseLevel = 100.0

    Seq(RegressionTreeLearner(), GuessTheMeanLearner()).foreach { baseLearner =>
      // These are in Seqs as a convenience for repurposing this test as a diagnostic tool.
      Seq(128).foreach { nRows =>
        Seq(16).foreach { nCols =>
          Seq(2).map { n => n * nRows }.foreach { nBags =>
            // Used for error output.
            val configDescription =
              s"learner=${baseLearner.getClass().toString()}\tnRows=$nRows\tnCols=$nCols\tnumBags=$nBags"

            val sigmaObsAndSigmaMean: Seq[(Double, Double)] = (1 to 20).flatMap { _ =>
              val trainingDataTmp =
                DataGenerator.generate(nRows, nCols, noise = 0.0, function = _ => 0.0, rng = rng).data
              val trainingData = trainingDataTmp.map { row =>
                row.mapLabel(_ + noiseLevel * rng.nextDouble())
              }
              val baggedLearner = RegressionBagger(baseLearner, numBags = nBags, uncertaintyCalibration = true)
              val RFMeta = baggedLearner.train(trainingData, rng = rng)
              val RF = RFMeta.model
              val results = RF.transform(trainingData.take(4).map(_.inputs))

              val sigmaMean: Seq[Double] = results.uncertainty(observational = false).get.asInstanceOf[Seq[Double]]
              sigmaMean.zip(results.asInstanceOf[RegressionResult].stdDevMean.get).foreach {
                case (a, b) =>
                  assert(a == b, s"Expected getUncertainty(observational=false)=getStdDevMean() for $configDescription")
              }

              val sigmaObs: Seq[Double] = results.uncertainty().get.asInstanceOf[Seq[Double]]
              sigmaObs.zip(results.asInstanceOf[RegressionResult].stdDevObs.get).foreach {
                case (a, b) =>
                  assert(a == b, s"Expected getUncertainty()=getStdDevObs() for $configDescription")
              }

              // We have strong theoretical guarantees on the behavior of GuessTheMeanLearner, so let's exercise them.
              // NOTE: these bounds reflect a ~3x systematic variance under-estimation in this particular test setting.
              {
                val rtolLower = baseLearner match {
                  case _: GuessTheMeanLearner => 3.5
                  case _: Any                 => 10.0
                }
                val rtolUpper = baseLearner match {
                  case _: GuessTheMeanLearner => 1.0
                  case _: Any                 => 1.0
                }
                sigmaObs.foreach { s =>
                  assert(
                    rtolLower * s > noiseLevel,
                    s"Observational StdDev getUncertainty() is too small for $configDescription"
                  )
                  assert(
                    s < rtolUpper * noiseLevel,
                    s"Observational StdDev getUncertainty() is too large for $configDescription"
                  )
                }
              }
              {
                val rtolLower = baseLearner match {
                  case _: GuessTheMeanLearner => 5.0
                  case _: Any                 => 1e3
                }
                val rtolUpper = baseLearner match {
                  case _: GuessTheMeanLearner => 1.0
                  case _: Any                 => 10.0
                }
                sigmaMean.foreach { s =>
                  assert(
                    rtolLower * s > (noiseLevel / Math.sqrt(nRows)),
                    s"Mean StdDev getUncertainty(observational=false)=$s is too small for $configDescription."
                  )
                  assert(
                    s < (rtolUpper * noiseLevel / Math.sqrt(nRows)),
                    s"Mean StdDev getUncertainty(observational=false)=$s is too large for $configDescription"
                  )
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
            val d = Beta(countSigmaObsGreater + 0.5, sigmaObsAndSigmaMean.length - countSigmaObsGreater + 0.5)(
              RandBasis.systemSeed
            )
            val minRateSigmaObsGreater = 0.9
            val level = 1e-4
            val probSigmaObsLess = d.cdf(minRateSigmaObsGreater)
            assert(
              probSigmaObsLess < level,
              s"Uncertainty should be greater when observational = true for $configDescription"
            )
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
  private def testConsistency(trainingData: Seq[TrainingRow[Double]], model: BaggedModel[Double]): Unit = {
    val testSubset = rng.shuffle(trainingData).take(16)
    val (singleValues, singleObsUnc, singleMeanUnc) = testSubset.map {
      case TrainingRow(x, _, _) =>
        val res = model.transform(Seq(x))
        (
          res.expected.head,
          res.uncertainty(true).get.head.asInstanceOf[Double],
          res.uncertainty(false).get.head.asInstanceOf[Double]
        )
    }.unzip3

    val (multiValues, multiObsUnc, multiMeanUnc) = {
      val res = model.transform(testSubset.map(_.inputs))
      (
        res.expected,
        res.uncertainty(true).get.map(_.asInstanceOf[Double]),
        res.uncertainty(false).get.map(_.asInstanceOf[Double])
      )
    }

    singleValues.zip(multiValues).zipWithIndex.foreach {
      case ((x, y), idx) =>
        assert(Math.abs(x - y) < 1.0e-9, s"Mean Uncertainty $x was not $y for $idx")
    }
    singleObsUnc.zip(multiObsUnc).zipWithIndex.foreach {
      case ((x, y), idx) =>
        assert(Math.abs(x - y) < 1.0e-9, s"Obs Uncertainty $x was not $y for $idx")
    }
    singleMeanUnc.zip(multiMeanUnc).zipWithIndex.foreach {
      case ((x, y), idx) =>
        assert(Math.abs(x - y) < 1.0e-9, s"Mean Uncertainty $x was not $y for $idx")
    }
  }
}
