package io.citrine.lolo.learners

import breeze.stats.distributions.{Beta, RandBasis}
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TrainingRow}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.random.Random
import org.junit.Test
import org.scalatest.Assertions._

@Test
class ExtraRandomTreesTest extends SeedRandomMixIn {

  /**
    * Test that the regression does the same thing as the regression bagger
    */
  @Test
  def testRegression(): Unit = {
    val trainingData = DataGenerator
      .generate(256, 5, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta = ExtraRandomTreesRegressor(randomlyRotateFeatures = randomlyRotateFeatures)
        .train(trainingData, rng = rng)
      val RF = RFMeta.model

      val loss = RFMeta.loss.get
      assert(0 <= loss && loss < 1e-8, "Expected zero loss.")

      val results = RF.transform(trainingData.map(_.inputs))
      // val means = results.getExpected()
      val sigma: Seq[Double] = results.uncertainty(observational = false).get.asInstanceOf[Seq[Double]]
      assert(sigma.forall(_ >= 0.0))

      assert(
        results.uncertainty(observational = true).isEmpty,
        "Observational uncertainty should be empty, since jackknife is disabled."
      )

      assert(results.gradient.isEmpty, "Returned a gradient when there shouldn't be one")

      if (!randomlyRotateFeatures) {
        /* The first feature should be the most important */
        val importances = RFMeta.featureImportance.get
        assert(importances(1) == importances.max)
      }
    }
  }

  /**
    * Test that the regression does the same thing as the classification bagger
    */
  @Test
  def testClassification(): Unit = {
    val nTrain = 64
    val nTest = nTrain
    val nBins = 8

    val trainingData = DataGenerator
      .generate(nTrain, 5, noise = 0.0, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedLabels(bins = nBins)
      .data

    /* Generate small perturbations from the training data */
    val testData = trainingData
      .map(_.mapInputs(_.map(x => x.asInstanceOf[Double] + 0.05 * rng.nextGaussian())))
      .take(nTest)

    Seq(true, false).foreach { randomlyRotateFeatures =>
      Seq(true, false).foreach { disableBootstrap =>
        Seq(1, 2).foreach { minLeafInstances =>
          rng = Random(238834L)

          val RFMeta = ExtraRandomTreesClassifier(
            numTrees = trainingData.size * 4,
            randomlyRotateFeatures = randomlyRotateFeatures,
            disableBootstrap = disableBootstrap,
            minLeafInstances = minLeafInstances
          ).train(trainingData, rng = rng)
          val RF = RFMeta.model

          /* Inspect the results on the training set */
          val trueRateTrainingSet = trainingData
            .map(_.label)
            .zip(RF.transform(trainingData.map(_.inputs)).expected)
            .count { case (a, p) => a == p }
            .toDouble / nTrain
          if (minLeafInstances == 1 && disableBootstrap) {
            assert(trueRateTrainingSet == 1.0)
          } else {
            assert(trueRateTrainingSet > 0.9)
          }

          /* Inspect the results on the perturbed test set */
          val results = RF.transform(testData.map(_.inputs))
          val means = results.expected
          val trueRateTestSet = testData.map(_.label).zip(means).count { case (a, p) => a == p }.toDouble / nTest
          assert(trueRateTestSet > 0.5)

          /* Check that class probabilities are reasonable */
          val uncertainty = results.uncertainty()
          assert(uncertainty.isDefined)
          assert(testData.map(_.label).zip(uncertainty.get).forall {
            case (a, probs) =>
              val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
              val maxProb = classProbabilities(a)
              maxProb > 1.0 / (2 * nBins) && maxProb < 1.0 && Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6
          })
        }
      }
    }
  }

  /**
    * Ensure classification isn't biased toward one or the other ordering of class labels
    * when there are duplicate inputs.
    */
  @Test
  def testClassificationUnbiased(): Unit = {
    val numTrials = 20
    val (winsSuffixed, winsPrefixed): (Int, Int) = (0 until numTrials)
      .map { _ =>
        val nTrain = 64
        val nTest = 16
        val (mainTrainingData, testData) = {
          val allData = DataGenerator
            .generate(
              nTrain + nTest,
              5,
              noise = 0.1,
              function = Friedman.friedmanSilverman,
              rng = rng
            )
            .withBinnedLabels(bins = 2)
            .data
          (allData.take(nTrain), allData.takeRight(nTest))
        }
        val dupeLabel = "DUPE"
        val trainingDataSuffixed = mainTrainingData ++ Seq(
          TrainingRow(mainTrainingData.head.inputs, dupeLabel)
        )
        val trainingDataPrefixed = Seq(
          TrainingRow(mainTrainingData.head.inputs, dupeLabel)
        ) ++ mainTrainingData

        val RFSuffixed =
          ExtraRandomTreesClassifier(numTrees = trainingDataSuffixed.size * 2).train(trainingDataSuffixed, rng = rng)
        val RFPrefixed =
          ExtraRandomTreesClassifier(numTrees = trainingDataPrefixed.size * 2).train(trainingDataPrefixed, rng = rng)
        val predictedSuffixed = RFSuffixed.model.transform(testData.map(_.inputs))
        val predictedPrefixed = RFPrefixed.model.transform(testData.map(_.inputs))
        val extraLabelCountSuffixed = predictedSuffixed.expected.count { case p: String => p == dupeLabel }
        val extraLabelCountPrefixed = predictedPrefixed.expected.count { case p: String => p == dupeLabel }

        if (extraLabelCountSuffixed > extraLabelCountPrefixed) {
          (1, 0)
        } else if (extraLabelCountSuffixed < extraLabelCountPrefixed) {
          (0, 1)
        } else {
          (0, 0)
        }
      }
      .asInstanceOf[Seq[(Int, Int)]]
      .reduce { (a: (Int, Int), b: (Int, Int)) => (a._1 + b._1, a._2 + b._2) }

    // Posterior beta distribution with Jeffreys prior.
    val d = Beta(winsSuffixed + 0.5, winsPrefixed + 0.5)(RandBasis.systemSeed)
    val l = d.inverseCdf(2e-6)
    val r = d.inverseCdf(1 - 2e-6)
    val tol = 1e-2
    assert(
      l < 0.5 - tol,
      f"Bias detected toward prefixed duplicate rows: rate ${d.mean}%.3f (1e-6 CI ${l}%.3f - ${r}%.3f) should be close to 0.5"
    )
    assert(
      r > 0.5 + tol,
      f"Bias detected toward suffixed duplicate rows: rate ${d.mean}%.3f (1e-6 CI ${l}%.3f - ${r}%.3f) should be close to 0.5"
    )
  }

  /**
    * Make sure that we can draw training weights consistently even when the training size is small
    */
  @Test
  def testWeightsWithSmallData(): Unit = {
    val trainingData = DataGenerator.generate(8, 1).data
    // the number of trees is the number of times we generate weights
    // so this has the effect of creating lots of different sets of weights
    val learner = ExtraRandomTreesRegressor(numTrees = 16384)
    // the test is that this training doesn't throw an exception
    learner.train(trainingData, rng = rng).model
  }

  /** Test that the same random seed leads to identical models */
  @Test
  def testReproducibility(): Unit = {
    val numTrain = 50
    // Generate completely random training data
    val realRows = DataGenerator.generate(rows = numTrain, cols = 12, noise = 5.0, function = _ => 0.0, rng = rng).data
    val catRows = realRows.map(_.withLabel(rng.nextBoolean()))

    // Generate test points
    val numTest = 25
    val testInputs =
      DataGenerator.generate(rows = numTest, cols = 12, function = _ => 0.0, rng = rng).data.map(_.inputs)

    val seed = 67852103L
    val extraRegressor = ExtraRandomTreesRegressor(
      biasLearner = Some(RegressionTreeLearner(maxDepth = 5)),
      randomlyRotateFeatures = true
    )
    val regModel1 = extraRegressor.train(realRows, rng = Random(seed)).model
    val regModel2 = extraRegressor.train(realRows, rng = Random(seed)).model
    val regPredictions1 = regModel1.transform(testInputs)
    val regPredictions2 = regModel2.transform(testInputs)
    assert(regPredictions1.expected == regPredictions2.expected)

    val extraClassifier = ExtraRandomTreesClassifier(randomlyRotateFeatures = true)
    val classModel1 = extraClassifier.train(catRows, rng = Random(seed)).model
    val classModel2 = extraClassifier.train(catRows, rng = Random(seed)).model
    val classPredictions1 = classModel1.transform(testInputs)
    val classPredictions2 = classModel2.transform(testInputs)
    assert(classPredictions1.expected == classPredictions2.expected)
  }
}
