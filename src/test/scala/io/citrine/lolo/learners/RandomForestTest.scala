package io.citrine.lolo.learners

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.{Beta, RandBasis}
import io.citrine.lolo.{DataGenerator, Learner, SeedRandomMixIn, TrainingRow}
import io.citrine.lolo.bags.MultiTaskBaggedPrediction
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.random.Random
import org.junit.Test
import org.scalatest.Assertions._

@Test
class RandomForestTest extends SeedRandomMixIn {

  /**
    * Test that the regression forest does the same thing as the regression bagger
    */
  @Test
  def testRegressionForest(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta = RandomForestRegressor(randomlyRotateFeatures = randomlyRotateFeatures)
        .train(trainingData)
      val RF = RFMeta.model

      assert(RFMeta.loss.get < 1.0, "Loss of bagger is larger than expected")

      val results = RF.transform(trainingData.map(_.inputs))
      // val means = results.expected
      val sigma: Seq[Double] = results.uncertainty().get.asInstanceOf[Seq[Double]]
      assert(sigma.forall(_ >= 0.0))

      assert(results.gradient.isEmpty, "Returned a gradient when there shouldn't be one")

      if (!randomlyRotateFeatures) {
        /* The first feature should be the most important */
        val importances = RFMeta.featureImportance.get
        assert(importances(1) == importances.max)
      }
    }
  }

  /** Test that a random forest with multiple outputs produces a multitask bagger. */
  @Test
  def testMultiTaskForest(): Unit = {
    val realRows = DataGenerator
      .generate(256, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    val inputs = realRows.map(_.inputs)
    val realLabels = realRows.map(_.label)
    val catLabels = realLabels.map(_ > realLabels.max / 2.0)
    val quadLabels = realLabels.map(x => x * x)
    val allLabels = Vector(realLabels, catLabels, quadLabels).transpose
    val multiTaskRows = TrainingRow.build(inputs.zip(allLabels))

    val RFMeta = MultiTaskRandomForest().train(multiTaskRows, rng = rng)
    val model = RFMeta.model

    val results = model.transform(inputs).asInstanceOf[MultiTaskBaggedPrediction]
    assert(results.uncertainty().isDefined)
    assert(results.uncertaintyCorrelation(0, 2).isDefined)
  }

  /**
    * Test that the regression forest does the same thing as the classification bagger
    */
  @Test
  def testClassificationForest(): Unit = {
    val trainingData = DataGenerator
      .generate(128, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .withBinnedLabels(bins = 8)
      .data

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta =
        RandomForestClassifier(numTrees = trainingData.size * 2, randomlyRotateFeatures = randomlyRotateFeatures)
          .train(trainingData, rng = rng)
      val RF = RFMeta.model

      /* Inspect the results */
      val results = RF.transform(trainingData.map(_.inputs))
      val means = results.expected
      assert(trainingData.map(_.label).zip(means).forall { case (a, p) => a == p })

      val uncertainty = results.uncertainty()
      assert(uncertainty.isDefined)
      assert(trainingData.map(_.label).zip(uncertainty.get).forall {
        case (a, probs) =>
          val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
          val maxProb = classProbabilities(a)
          maxProb > 0.5 && maxProb < 1.0 && Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6
      })
    }
  }

  /** Test that the same random seed leads to identical models */
  @Test
  def testReproducibility(): Unit = {
    def checkReproducibility[T](
        learner: Learner[T],
        trainingData: Seq[TrainingRow[T]],
        testInputs: Seq[Vector[Any]],
        seed: Long
    ): Unit = {
      val model1 = learner.train(trainingData, rng = Random(seed)).model
      val model2 = learner.train(trainingData, rng = Random(seed)).model
      val predictions1 = model1.transform(testInputs)
      val predictions2 = model2.transform(testInputs)
      assert(predictions1.expected == predictions2.expected)
    }

    val numTrain = 50
    // Generate completely random training data
    val realRows = DataGenerator
      .generate(rows = numTrain, cols = 12, noise = 5.0, function = _ => 0.0, rng = rng)
      .data
    val catRows = realRows.map(_.withLabel(rng.nextBoolean()))
    val multiTaskRows = realRows.zip(catRows).map {
      case (r, c) => r.withLabel(Vector(r.label, c.label))
    }

    // Generate test points
    val numTest = 25
    val testInputs =
      DataGenerator.generate(rows = numTest, cols = 12, function = _ => 0.0, rng = rng).data.map(_.inputs)

    val seed = 67852103L
    val rfRegressor = RandomForestRegressor(
      biasLearner = Some(RegressionTreeLearner(maxDepth = 5)),
      randomizePivotLocation = true,
      randomlyRotateFeatures = true
    )
    checkReproducibility(rfRegressor, realRows, testInputs, seed)

    val rfClassifier = RandomForestClassifier(
      randomizePivotLocation = true,
      randomlyRotateFeatures = true
    )
    checkReproducibility(rfClassifier, catRows, testInputs, seed)

    val rfMultiTask = MultiTaskRandomForest(
      randomizePivotLocation = true,
      randomlyRotateFeatures = true
    )
    checkReproducibility(rfMultiTask, multiTaskRows, testInputs, seed)
  }

  /**
    * Ensure classification forest isn't biased toward one or the other ordering of class labels
    * when there are duplicate inputs.
    */
  @Test
  def testClassificationForestUnbiased(): Unit = {
    val numTrials = 20
    val (winsSuffixed, winsPrefixed): (Int, Int) = (0 until numTrials)
      .map { _ =>
        val mainTrainingData = DataGenerator
          .generate(64, 5, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
          .withBinnedLabels(bins = 2)
          .data

        val dupeLabel = "DUPE"
        val trainingDataSuffixed = mainTrainingData ++ Seq(
          TrainingRow(mainTrainingData.head.inputs, dupeLabel)
        )
        val trainingDataPrefixed = Seq(
          TrainingRow(mainTrainingData.head.inputs, dupeLabel)
        ) ++ mainTrainingData

        val RFSuffixed =
          RandomForestClassifier(numTrees = trainingDataSuffixed.size * 2).train(trainingDataSuffixed, rng = rng)
        val RFPrefixed =
          RandomForestClassifier(numTrees = trainingDataPrefixed.size * 2).train(trainingDataPrefixed, rng = rng)
        val predictedSuffixed = RFSuffixed.model.transform(mainTrainingData.map(_.inputs))
        val predictedPrefixed = RFPrefixed.model.transform(mainTrainingData.map(_.inputs))
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
    * Randomized splits should do really well on linear signals when there are lots of trees.  Test that they
    * outperform mid-point splits
    */
  @Test
  def testRandomizedSplitLocations(): Unit = {
    // Generate a linear signal in one dimension: 2 * x
    val trainingData = DataGenerator
      .generate(32, 1, function = { x => x.head * 2.0 }, rng = rng)
      .data

    // Create a consistent set of parameters
    val baseForest = RandomForestRegressor(numTrees = 16384, useJackknife = false)

    // Turn off split randomization and compute the loss (out-of-bag error)
    val lossWithoutRandomization: Double = baseForest
      .copy(randomizePivotLocation = false)
      .train(trainingData, rng = rng)
      .loss
      .get

    // Turn on split randomization and compute the loss (out-of-bag error)
    val lossWithRandomization: Double = baseForest
      .copy(randomizePivotLocation = true)
      .train(trainingData, rng = rng)
      .loss
      .get

    assert(lossWithRandomization < lossWithoutRandomization)
  }

  /**
    * Make sure that we can draw training weights consistently even when the training size is small
    */
  @Test
  def testWeightsWithSmallData(): Unit = {
    val trainingData = DataGenerator.generate(8, 1, rng = rng).data
    // the number of trees is the number of times we generate weights
    // so this has the effect of creating lots of different sets of weights
    val learner = RandomForestRegressor(numTrees = 16384)
    // the test is that this training doesn't throw an exception
    learner.train(trainingData, rng = rng).model
  }

  def shapleyCompare(
      trainingData: Seq[TrainingRow[Double]],
      evalLocation: Vector[Any],
      expected: Vector[Double],
      rtol: Double = 0.1
  ): Unit = {
    val actual = RandomForestRegressor().train(trainingData, rng = rng).model.shapley(evalLocation) match {
      case None => fail("Unexpected None returned by shapley.")
      case x: Option[DenseMatrix[Double]] => {
        val a = x.get
        assert(a.cols == trainingData.head.inputs.length, "Expected one Shapley value per feature.")
        assert(a.rows == 1, "Expected a single output dimension.")
        a.toDenseVector.toScalaVector
      }
      case _ => fail("Unexpected return type.")
    }
    expected.zip(actual).foreach {
      case (e: Double, a: Double) => assert(Math.abs(e - a) / a < rtol)
    }
  }

  /**
    * Test Shapley value for  a simple tree.
    */
  @Test
  def testShapley(): Unit = {
    rng = Random(3753L)

    // Example from Lundberg paper (https://arxiv.org/pdf/1802.03888.pdf)
    val trainingData1 = Seq(
      TrainingRow(Vector(1.0, 1.0), 80.0),
      TrainingRow(Vector(1.0, 0.0), 0.0),
      TrainingRow(Vector(0.0, 1.0), 0.0),
      TrainingRow(Vector(0.0, 0.0), 0.0)
    )
    val expected1 = Vector(30.0, 30.0)
    shapleyCompare((1 to 8).flatMap(_ => trainingData1), Vector[Any](1.0, 1.0), expected1)

    // Second example from Lundberg paper (https://arxiv.org/pdf/1802.03888.pdf)
    val trainingData2 = Seq(
      TrainingRow(Vector(1.0, 1.0), 90.0),
      TrainingRow(Vector(1.0, 0.0), 10.0),
      TrainingRow(Vector(0.0, 1.0), 0.0),
      TrainingRow(Vector(0.0, 0.0), 0.0)
    )
    val expected2 = Vector(35.0, 30.0)
    shapleyCompare((1 to 8).flatMap(_ => trainingData2), Vector[Any](1.0, 1.0), expected2)

    // Example with two splits on one feature
    // Worked out with pen-and-paper from Lundberg Equation 2.
    val trainingData3 = Seq(
      TrainingRow(Vector(1.0, 1.0), 100.0),
      TrainingRow(Vector(1.0, 0.0), 80.0),
      TrainingRow(Vector(1.0, 0.2), 70.0),
      TrainingRow(Vector(0.0, 1.0), 0.0),
      TrainingRow(Vector(0.0, 0.2), 0.0),
      TrainingRow(Vector(0.0, 0.0), 0.0)
    )
    val expected3 = Vector(45.8333333333333, 12.5)
    shapleyCompare((1 to 8).flatMap(_ => trainingData3), Vector[Any](1.0, 1.0), expected3)

    // Example with 5 features, to exercise all the factorials in Lundberg equation 2.
    // Referenced against the shap package on a sklearn decision tree.
    val trainingData4 = Seq(
      TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 0.0), 1.0),
      TrainingRow(Vector(1.0, 0.0, 0.0, 0.0, 0.0), 2.0),
      TrainingRow(Vector(0.0, 1.0, 0.0, 0.0, 0.0), 4.0),
      TrainingRow(Vector(0.0, 0.0, 1.0, 0.0, 0.0), 8.0),
      TrainingRow(Vector(0.0, 0.0, 0.0, 1.0, 0.0), 16.0),
      TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 1.0), 32.0)
    )
    val expected4 = Vector(0.0333333333333333, 0.2, 0.8666666666666667, 3.533333333333333, 16.866666666666667)
    shapleyCompare((1 to 8).flatMap { _ => trainingData4 }, Vector.fill[Any](5)(1.0), expected4)
  }
}
