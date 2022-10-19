package io.citrine.lolo.bags

import breeze.stats.distributions.{Beta, RandBasis}
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TrainingRow}
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

@Test
class MultiTaskBaggerTest extends SeedRandomMixIn {

  /**
    * Test that we get a reasonable output on a single regression problem
    */
  @Test
  def testSingleRegression(): Unit = {
    val trainingData = DataGenerator
      .generate(512, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    val reshapedTrainingData = trainingData.map(_.mapLabel(Vector(_)))
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(
      DTLearner,
      numBags = trainingData.size,
      uncertaintyCalibration = true
    )
    val RFMeta = baggedLearner.train(reshapedTrainingData, rng = rng)
    val RF = RFMeta.models.head

    val results = RF.transform(trainingData.map(_.inputs))
    val means = results.expected
    val sigma: Seq[Double] = results.uncertainty().get.asInstanceOf[Seq[Double]]
    assert(sigma.forall(_ >= 0.0))
    assert(
      results.asInstanceOf[MultiPointBaggedPrediction].rescaleRatio != 1.0,
      "uncertainty calibration ratio was not included in prediction result"
    )
    assert(results.gradient.isEmpty, "Returned a gradient when there shouldn't be one")
    assert(RFMeta.loss.get < 1.0, "Loss of bagger is larger than expected")
  }

  /**
    * Test UQ on multitask regression with a single regression problem.
    */
  @Test
  def testBaggedMultiTaskGetUncertainty(): Unit = {
    val noiseLevel = 100.0

    Seq(MultiTaskTreeLearner()).foreach { baseLearner =>
      // These are in Seqs as a convenience for repurposing this test as a diagnostic tool.
      Seq(128).foreach { nRows =>
        Seq(16).foreach { nCols =>
          Seq(2).map { n => n * nRows }.foreach { nBags =>
            // Used for error output.
            val configDescription =
              s"learner=${baseLearner.getClass.toString}\tnRows=$nRows\tnCols=$nCols\tnumBags=$nBags"

            val sigmaObsAndSigmaMean: Seq[(Double, Double)] = (1 to 20).flatMap { _ =>
              val trainingDataTmp = DataGenerator.generate(nRows, nCols, function = _ => 0.0, rng = rng).data
              val trainingData = trainingDataTmp.map(_.mapLabel(_ + noiseLevel * rng.nextDouble()))

              val reshapedTrainingData = trainingData.map(_.mapLabel(Vector(_)))
              val baggedLearner = MultiTaskBagger(
                baseLearner,
                numBags = nBags,
                uncertaintyCalibration = true
              )
              val RFMeta = baggedLearner.train(reshapedTrainingData, rng = rng)
              val RF = RFMeta.models.head
              val results = RF.transform(trainingData.take(4).map(_.inputs))

              val sigmaMean: Seq[Double] = results.uncertainty(observational = false).get.asInstanceOf[Seq[Double]]
              sigmaMean.zip(results.asInstanceOf[MultiPointBaggedPrediction].stdDevMean.get).foreach {
                case (a, b) =>
                  assert(a == b, s"Expected getUncertainty(observational=false)=getStdDevMean() for $configDescription")
              }

              val sigmaObs: Seq[Double] = results.uncertainty().get.asInstanceOf[Seq[Double]]
              sigmaObs.zip(results.asInstanceOf[MultiPointBaggedPrediction].stdDevObs.get).foreach {
                case (a, b) =>
                  assert(a == b, s"Expected getUncertainty()=getStdDevObs() for $configDescription")
              }

              // NOTE: these bounds reflect a ~3x systematic variance under-estimation in this particular test setting.
              {
                val rtolLower = baseLearner match {
                  case _: MultiTaskTreeLearner => 10.0
                  case _: Any                  => fail("Not implemented.")
                }
                val rtolUpper = baseLearner match {
                  case _: MultiTaskTreeLearner => 1.0
                  case _: Any                  => fail("Not implemented.")
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
                  case _: MultiTaskTreeLearner => 1e3
                  case _: Any                  => fail("Not implemented.")
                }
                val rtolUpper = baseLearner match {
                  case _: MultiTaskTreeLearner => 10.0
                  case _: Any                  => fail("Not implemented.")
                }
                sigmaMean.foreach { s =>
                  assert(
                    rtolLower * s > (noiseLevel / Math.sqrt(nRows)),
                    s"Mean StdDev getUncertainty(observational=false) is too small for $configDescription."
                  )
                  assert(
                    s < (rtolUpper * noiseLevel / Math.sqrt(nRows)),
                    s"Mean StdDev getUncertainty(observational=false) is too large for $configDescription"
                  )
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
    * Test the we get a reasonable result on a single classification problem
    */
  @Test
  def testClassificationBagger(): Unit = {
    val trainingData = DataGenerator
      .generate(128, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .withBinnedLabels(bins = 8)
      .data

    val reshapedTrainingData = trainingData.map(_.mapLabel(Vector(_)))
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner =
      MultiTaskBagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(reshapedTrainingData, rng = rng)
    val RF = RFMeta.models.head

    /* Inspect the results */
    val results = RF.transform(trainingData.map(_.inputs))
    val means = results.expected
    assert(trainingData.map(_.label).zip(means).forall { case (a, p) => a == p })

    val uncertainty = results.uncertainty()
    assert(uncertainty.isDefined)
    trainingData.map(_.label).zip(uncertainty.get).foreach {
      case (a, probs) =>
        val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
        val maxProb = classProbabilities(a)
        assert(maxProb >= 0.5)
        assert(maxProb < 1.0)
        assert(Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6)
    }
    assert(results.gradient.isEmpty, "Returned a gradient when there shouldn't be one")
  }

  /** Test that a multi-task bagged model properly stores and transposes individual trees, and remembers labels. */
  @Test
  def testCombinedMultiTaskModel(): Unit = {
    val numTrain = 256
    val numBags = 64
    val numTest = 32

    val realRows = DataGenerator
      .generate(numTrain, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .data
    val (inputs, realLabel) = realRows.map(row => (row.inputs, row.label)).unzip
    val catLabel = realLabel.map(_ > realLabel.max / 2.0)
    val labels = Vector(realLabel, catLabel).transpose
    val multiTaskRows = TrainingRow.build(inputs.zip(labels))

    val learner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags = numBags,
      biasLearner = Some(RegressionTreeLearner(maxDepth = 2)),
      uncertaintyCalibration = true
    )
    val RF = baggedLearner.train(multiTaskRows, rng = rng)

    val testInputs = inputs.take(numTest)
    val predictionResult = RF.model.transform(testInputs)
    assert(predictionResult.ensemblePredictions.length == numBags)

    // because the uncertainty is recalibrated, the prediction result should have a rescale value that is not equal to 1.0
    assert(predictionResult.labelPredictions.head.asInstanceOf[MultiPointBaggedPrediction].rescaleRatio != 1.0)

    // The prediction made by the full model and the prediction made by just the categorical model should agree
    // and both be equal to the training label.
    val expected = predictionResult.expected
    val expectedCat = RF.models(1).transform(testInputs).expected
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

    val realRows = DataGenerator
      .generate(numTrain, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .data
    val (inputs, realLabel) = realRows.map(row => (row.inputs, row.label)).unzip
    val catLabel = realLabel.map(_ > realLabel.max / 2.0)
    val correlatedLabel = DataGenerator.makeLinearlyCorrelatedData(realLabel, trainingRho, rng = rng)
    val labels = Vector(realLabel, catLabel, correlatedLabel).transpose
    val multiTaskRows = TrainingRow.build(inputs.zip(labels))

    val learner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags = numBags,
      biasLearner = Some(RegressionTreeLearner(maxDepth = 2))
    )
    val RF = baggedLearner.train(multiTaskRows, rng = rng).model

    val testInputs =
      DataGenerator.generate(numTest, 12, function = Friedman.friedmanSilverman, rng = rng).data.map(_.inputs)
    val predictionResult = RF.transform(testInputs)

    Seq(true, false).foreach { observational =>
      // All real-valued predictions should be perfectly correlated with themselves
      assert(predictionResult.uncertaintyCorrelation(0, 0, observational).get == Seq.fill(numTest)(1.0))
      // Correlation with a non-real-valued label should be empty
      assert(predictionResult.uncertaintyCorrelation(0, 1, observational).isEmpty)
      // Otherwise, all we can assert is that -1.0 <= rho <= 1.0
      predictionResult.uncertaintyCorrelation(0, 2, observational).get.foreach { calcRho =>
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
    val realRows = DataGenerator.generate(256, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
    val inputs = realRows.map(_.inputs)
    val realLabel: Seq[Double] = realRows.map(_.label)
    val catLabel: Seq[Boolean] = realRows.map(_.label > realLabel.max / 2.0)
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
    val labels = Vector(sparseReal, sparseCat).transpose
    val multiTaskRows = TrainingRow.build(inputs.zip(labels))

    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(
      DTLearner,
      numBags = inputs.size,
      biasLearner = Some(GuessTheMeanLearner())
    )
    val trainingResult = baggedLearner.train(multiTaskRows, rng = rng)
    val RF = trainingResult.models.last

    val catResults = RF.transform(inputs).expected
    val realUncertainty = trainingResult.models.head.transform(inputs).uncertainty().get
    assert(realUncertainty.forall(!_.asInstanceOf[Double].isNaN), s"Some uncertainty values were NaN")

    val referenceRows = TrainingRow.build(inputs.zip(sparseCat)).filterNot(_.label == null)
    val referenceModel = ClassificationBagger(ClassificationTreeLearner(), numBags = inputs.size)
      .train(referenceRows, rng = rng)
    val reference = referenceModel
      .model
      .transform(inputs)
      .expected

    val singleF1 = ClassificationMetrics.f1scores(reference, catLabel)
    val multiF1 = ClassificationMetrics.f1scores(catResults, catLabel)

    // Make sure we can grab the loss without issue
    assert(!referenceModel.loss.get.isNaN, "Single task classification loss was NaN")
    assert(!trainingResult.loss.get.isNaN, "Sparse multitask loss was NaN")

    assert(multiF1 > singleF1, s"Multi-task is under-performing single-task")
    assert(multiF1 <= 1.0, "Multitask classification F1 score was greater than 1.0")
  }

  /**
    * Test that a multitask model can train when there is no overlap between the outputs.
    */
  @Test
  def testFullSparsity(): Unit = {
    /* Setup some data */
    val realRows = DataGenerator.generate(256, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
    val inputs = realRows.map(_.inputs)
    val realLabel: Seq[Double] = realRows.map(_.label)
    val catLabel: Seq[Boolean] = realRows.map(_.label > realLabel.max / 2.0)

    // Test 2 real outputs, 2 categorical outputs, and an even split
    Seq(2, 128, 254).foreach { cutoffIndex =>
      val sparseCat = catLabel.zipWithIndex.map {
        case (x, idx) =>
          if (idx < cutoffIndex) null else x
      }
      val sparseReal = realLabel.zipWithIndex.map {
        case (x, idx) =>
          if (idx >= cutoffIndex) Double.NaN else x
      }
      val labels = Vector(sparseReal, sparseCat).transpose
      val multiTaskRows = TrainingRow.build(inputs.zip(labels))

      val baggedLearner = MultiTaskBagger(
        MultiTaskTreeLearner(),
        numBags = inputs.size,
        biasLearner = Some(GuessTheMeanLearner())
      )
      assert(baggedLearner.train(multiTaskRows, rng = rng).models.size == 2)
    }
  }
}
