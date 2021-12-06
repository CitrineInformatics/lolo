package io.citrine.lolo.bags

import breeze.stats.distributions.Beta
import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class MultiTaskBaggerTest {
  val rng = new Random(37895L)

  /**
    * Test that we get a reasonable output on a single regression problem
    */
  @Test
  def testSingleRegression(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(512, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = trainingData.size, randBasis = TestUtils.getBreezeRandBasis(10478L))
    val RFMeta = baggedLearner.train(inputs, Seq(labels))
    val RF = RFMeta.getModels().head


    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
    assert(sigma.forall(_ >= 0.0))

    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")
    assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")
  }

  /**
    * Test UQ on multitask regression with a single regression problem.
    */
  @Test
  def testBaggedMultiTaskGetUncertainty(): Unit = {
    val noiseLevel = 100.0

    Seq(MultiTaskTreeLearner()).foreach { baseLearner =>
      // These are in Seqs as a convenience for repurposing this test as a diagnostic tool.
      Seq(64).foreach { nRows =>
        Seq(16).foreach { nCols =>
          Seq(2).map { n => n * nRows }.foreach { nBags =>
            // Used for error output.
            val configDescription =s"learner=${baseLearner.getClass().toString()}\tnRows=$nRows\tnCols=$nCols\tnumBags=$nBags"

            val sigmaObsAndSigmaMean: Seq[(Double, Double)] = (1 to 20).flatMap { _ =>
              val trainingDataTmp = TestUtils.generateTrainingData(nRows, nCols, noise = 0.0, function = _ => 0.0, seed = rng.nextLong())
              val trainingData = trainingDataTmp.map { x => (x._1, x._2 + noiseLevel * rng.nextDouble()) }
              val inputs = trainingData.map(_._1)
              val labels = trainingData.map(_._2)
              val baggedLearner = MultiTaskBagger(baseLearner, numBags = nBags, uncertaintyCalibration = true, randBasis = TestUtils.getBreezeRandBasis(7835178L))
              val RFMeta = baggedLearner.train(inputs, Seq(labels))
              val RF = RFMeta.getModels().head
              val results = RF.transform(trainingData.take(4).map(_._1))

              val sigmaMean: Seq[Double] = results.getUncertainty(observational = false).get.asInstanceOf[Seq[Double]]
              sigmaMean.zip(results.asInstanceOf[MultiPredictionBaggedResult].getStdDevMean().get).foreach { case (a, b) =>
                assert(a == b, s"Expected getUncertainty(observational=false)=getStdDevMean() for $configDescription")
              }

              val sigmaObs: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
              sigmaObs.zip(results.asInstanceOf[MultiPredictionBaggedResult].getStdDevObs().get).foreach { case (a, b) =>
                assert(a == b, s"Expected getUncertainty()=getStdDevObs() for $configDescription")
              }

              // NOTE: these bounds reflect a ~3x systematic variance under-estimation in this particular test setting.
              {
                val rtolLower = baseLearner match {
                  case _: MultiTaskTreeLearner => 10.0
                  case _: Any => fail("Not implemented.")
                }
                val rtolUpper = baseLearner match {
                  case _: MultiTaskTreeLearner => 1.0
                  case _: Any => fail("Not implemented.")
                }
                sigmaObs.foreach { s =>
                  assert(rtolLower * s > noiseLevel, s"Observational StdDev getUncertainty() is too small for $configDescription")
                  assert(s < rtolUpper * noiseLevel, s"Observational StdDev getUncertainty() is too large for $configDescription")
                }
              }
              {
                val rtolLower = baseLearner match {
                  case _: MultiTaskTreeLearner => 1e3
                  case _: Any => fail("Not implemented.")
                }
                val rtolUpper = baseLearner match {
                  case _: MultiTaskTreeLearner => 10.0
                  case _: Any => fail("Not implemented.")
                }
                sigmaMean.foreach { s =>
                  assert(rtolLower * s > (noiseLevel / Math.sqrt(nRows)), s"Mean StdDev getUncertainty(observational=false) is too small for $configDescription.")
                  assert(s < (rtolUpper * noiseLevel / Math.sqrt(nRows)), s"Mean StdDev getUncertainty(observational=false) is too large for $configDescription")
                }

                // Uncomment for diagnostic output.
                // sigmaObs.zip(sigmaMean).foreach { case (sObs, sMean) =>
                //   println(s"$configDescription\tsObs=$sObs\tsMean=$sMean")
                // }
              }

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
    * Test the we get a reasonable result on a single classification problem
    */
  @Test
  def testClassificationBagger(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(128, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = trainingData.size, randBasis = TestUtils.getBreezeRandBasis(478L))
    val RFMeta = baggedLearner.train(inputs, Seq(labels))
    val RF = RFMeta.getModels().head

    /* Inspect the results */
    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    assert(trainingData.map(_._2).zip(means).forall { case (a, p) => a == p })

    val uncertainty = results.getUncertainty()
    assert(uncertainty.isDefined)
    trainingData.map(_._2).zip(uncertainty.get).foreach { case (a, probs) =>
      val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
      val maxProb = classProbabilities(a)
      assert(maxProb >= 0.5)
      assert(maxProb < 1.0)
      assert(Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6)
    }
    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")
  }


  /** Test that a multi-task bagged model properly stores and transposes individual trees, and remembers labels. */
  @Test
  def testCombinedMultiTaskModel(): Unit = {
    val numTrain = 256
    val numBags = 64
    val numTest = 32
    val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(numTrain, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val inputs: Seq[Vector[Double]] = raw.map(_._1)
    val realLabel: Seq[Double] = raw.map(_._2)
    val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)

    val learner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags = numBags,
      biasLearner = Some(RegressionTreeLearner(maxDepth = 2)),
      randBasis = TestUtils.getBreezeRandBasis(78495L)
    )
    val RF = baggedLearner.train(inputs, Seq(realLabel, catLabel))

    val testInputs = inputs.take(numTest)
    val predictionResult = RF.getModel().transform(testInputs)
    assert(predictionResult.predictions.length == numBags)

    // The prediction made by the full model and the prediction made by just the categorical model should agree
    // and both be equal to the training label.
    val expected = predictionResult.getExpected()
    val expectedCat = RF.getModels()(1).transform(testInputs).getExpected()
    (0 until numTest).foreach { i =>
      assert(expected(i)(1) == catLabel(i))
      assert(catLabel(i) == expectedCat(i))
    }
  }

  /** Test the correlation coefficient calculation. */
  @Test
  def testCorrelation(): Unit = {
    val numTrain = 256
    val numBags = 64
    val numTest = 32
    val trainingRho = 0.45 // desired correlation between two real-valued training labels
    val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(numTrain, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val inputs: Seq[Vector[Double]] = raw.map(_._1)
    val realLabel: Seq[Double] = raw.map(_._2)
    val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)
    val correlatedLabel: Seq[Double] = TestUtils.makeLinearCorrelatedData(realLabel, trainingRho)

    val learner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags = numBags,
      biasLearner = Some(RegressionTreeLearner(maxDepth = 2)),
      randBasis = TestUtils.getBreezeRandBasis(78495L)
    )
    val RF = baggedLearner.train(inputs, Seq(realLabel, catLabel, correlatedLabel)).getModel()

    val testInputs = TestUtils.generateTrainingData(numTest, 12, function = Friedman.friedmanSilverman).map(_._1)
    val predictionResult = RF.transform(testInputs)

    Seq(true, false).foreach { observational =>
      // All real-valued predictions should be perfectly correlated with themselves
      assert(predictionResult.getUncertaintyCorrelation(0, 0, observational).get == Seq.fill(numTest)(1.0))
      // Correlation with a non-real-valued label should be empty
      assert(predictionResult.getUncertaintyCorrelation(0, 1, observational).isEmpty)
      // Otherwise, all we can assert is that -1.0 <= rho <= 1.0
      predictionResult.getUncertaintyCorrelation(0, 2, observational).get.foreach { calcRho =>
        assert(calcRho >= -1.0 && calcRho <= 1.0)
      }
    }

  }

  /**
    * Test that multi-task (regression, classification) with sparse classification labels
    * outperforms a direct classification model on the same label
    */
  @Test
  def testSparseMixedBagged(): Unit = {
    /* Setup some data */
    val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(256, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val inputs: Seq[Vector[Double]] = raw.map(_._1)
    val realLabel: Seq[Double] = raw.map(_._2)
    val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)
    val sparseCat = catLabel.map(x =>
      if (rng.nextDouble() > 0.125) {
        null
      } else {
        x
      }
    )
    val sparseReal = realLabel.map(x =>
      if (rng.nextDouble() > 0.5) {
        Double.NaN
      } else {
        x
      }
    )

    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = inputs.size, biasLearner = Some(GuessTheMeanLearner(rng = rng)), randBasis = TestUtils.getBreezeRandBasis(7839L))
    val trainingResult = baggedLearner.train(inputs, Seq(sparseReal, sparseCat))
    val RF = trainingResult.getModels().last

    val catResults = RF.transform(inputs).getExpected()
    val realUncertainty = trainingResult.getModels().head.transform(inputs).getUncertainty().get
    assert(realUncertainty.forall(!_.asInstanceOf[Double].isNaN), s"Some uncertainty values were NaN")

    val referenceModel = Bagger(ClassificationTreeLearner(), numBags = inputs.size)
      .train(inputs.zip(sparseCat).filterNot(_._2 == null))
    val reference = referenceModel
      .getModel()
      .transform(inputs)
      .getExpected()

    val singleF1 = ClassificationMetrics.f1scores(reference, catLabel)
    val multiF1 = ClassificationMetrics.f1scores(catResults, catLabel)

    // Make sure we can grab the loss without issue
    assert(!referenceModel.getLoss().get.isNaN, "Single task classification loss was NaN")
    assert(!trainingResult.getLoss().get.isNaN, "Sparse multitask loss was NaN")

    assert(multiF1 > singleF1, s"Multi-task is under-performing single-task")
    assert(multiF1 <= 1.0, "Multitask classification F1 score was greater than 1.0")
  }
}

