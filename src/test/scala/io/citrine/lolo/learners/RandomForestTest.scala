package io.citrine.lolo.learners

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.{Beta, RandBasis}
import io.citrine.lolo.{SeedRandomMixIn, TestUtils}
import io.citrine.lolo.bags.MultiTaskBaggedResult
import io.citrine.lolo.stats.functions.Friedman
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
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng),
      inputBins = Seq((0, 8))
    )

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta = RandomForest(randomlyRotateFeatures = randomlyRotateFeatures)
        .train(trainingData)
      val RF = RFMeta.getModel()

      assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

      val results = RF.transform(trainingData.map(_._1))
      // val means = results.getExpected()
      val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
      assert(sigma.forall(_ >= 0.0))

      assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")

      if (!randomlyRotateFeatures) {
        /* The first feature should be the most important */
        val importances = RFMeta.getFeatureImportance().get
        assert(importances(1) == importances.max)
      }
    }
  }

  /** Test that a random forest with multiple outputs produces a multitask bagger. */
  @Test
  def testMultitaskForest(): Unit = {
    val (inputs: Seq[Vector[Double]], realLabel: Seq[Double]) = TestUtils
      .binTrainingData(
        TestUtils
          .generateTrainingData(256, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng),
        inputBins = Seq((0, 8))
      )
      .unzip
    val catLabel: Seq[Boolean] = realLabel.map(_ > realLabel.max / 2.0)
    val quadLabel: Seq[Double] = realLabel.map(x => x * x)
    val labels = Vector(realLabel, catLabel, quadLabel).transpose

    val RFMeta = RandomForest().train(inputs.zip(labels), rng = rng)
    val model = RFMeta.getModel()

    val results = model.transform(inputs).asInstanceOf[MultiTaskBaggedResult]
    assert(results.getUncertainty().isDefined)
    assert(results.getUncertaintyCorrelation(0, 2).isDefined)
  }

  /**
    * Test that the regression forest does the same thing as the classification bagger
    */
  @Test
  def testClassificationForest(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils
        .generateTrainingData(128, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng),
      inputBins = Seq((0, 8)),
      responseBins = Some(8)
    )

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta =
        RandomForest(numTrees = trainingData.size * 2, randomlyRotateFeatures = randomlyRotateFeatures)
          .train(trainingData, rng = rng)
      val RF = RFMeta.getModel()

      /* Inspect the results */
      val results = RF.transform(trainingData.map(_._1))
      val means = results.getExpected()
      assert(trainingData.map(_._2).zip(means).forall { case (a, p) => a == p })

      val uncertainty = results.getUncertainty()
      assert(uncertainty.isDefined)
      assert(trainingData.map(_._2).zip(uncertainty.get).forall {
        case (a, probs) =>
          val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
          val maxProb = classProbabilities(a)
          maxProb > 0.5 && maxProb < 1.0 && Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6
      })
    }
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
        val mainTrainingData = TestUtils.binTrainingData(
          TestUtils
            .generateTrainingData(64, 5, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng),
          responseBins = Some(2)
        )
        val dupeLabel = "DUPE"
        val trainingDataSuffixed = mainTrainingData ++ Seq(
          (mainTrainingData.head._1, dupeLabel)
        )
        val trainingDataPrefixed = Seq(
          (mainTrainingData.head._1, dupeLabel)
        ) ++ mainTrainingData

        val RFSuffixed = RandomForest(numTrees = trainingDataSuffixed.size * 2).train(trainingDataSuffixed, rng = rng)
        val RFPrefixed = RandomForest(numTrees = trainingDataPrefixed.size * 2).train(trainingDataPrefixed, rng = rng)
        val predictedSuffixed = RFSuffixed.getModel().transform(mainTrainingData.map(_._1))
        val predictedPrefixed = RFPrefixed.getModel().transform(mainTrainingData.map(_._1))
        val extraLabelCountSuffixed = predictedSuffixed.getExpected().count { case p: String => p == dupeLabel }
        val extraLabelCountPrefixed = predictedPrefixed.getExpected().count { case p: String => p == dupeLabel }

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
    val trainingData: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(
      32,
      1,
      function = { x => x.head * 2.0 },
      rng = rng
    )

    // Create a consistent set of parameters
    val baseForest = RandomForest(numTrees = 16384, useJackknife = false)

    // Turn off split randomization and compute the loss (out-of-bag error)
    val lossWithoutRandomization: Double = baseForest
      .copy(randomizePivotLocation = false)
      .train(trainingData, rng = rng)
      .getLoss()
      .get

    // Turn on split randomization and compute the loss (out-of-bag error)
    val lossWithRandomization: Double = baseForest
      .copy(randomizePivotLocation = true)
      .train(trainingData, rng = rng)
      .getLoss()
      .get

    assert(lossWithRandomization < lossWithoutRandomization)
  }

  /**
    * Make sure that we can draw training weights consistently even when the training size is small
    */
  @Test
  def testWeightsWithSmallData(): Unit = {
    val trainingData = TestUtils.generateTrainingData(8, 1, rng = rng)
    // the number of trees is the number of times we generate weights
    // so this has the effect of creating lots of different sets of weights
    val learner = RandomForest(numTrees = 16384)
    // the test is that this training doesn't throw an exception
    learner.train(trainingData, rng = rng).getModel()
  }

  def shapleyCompare(
      trainingData: Seq[(Vector[Any], Double)],
      evalLocation: Vector[Any],
      expected: Vector[Double],
      rtol: Double = 5e-2
  ): Unit = {
    val actual = RandomForest().train(trainingData, rng = rng).getModel().shapley(evalLocation) match {
      case None => fail("Unexpected None returned by shapley.")
      case x: Option[DenseMatrix[Double]] => {
        val a = x.get
        assert(a.cols == trainingData.head._1.length, "Expected one Shapley value per feature.")
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
      (Vector(1.0, 1.0), 80.0),
      (Vector(1.0, 0.0), 0.0),
      (Vector(0.0, 1.0), 0.0),
      (Vector(0.0, 0.0), 0.0)
    )
    val expected1 = Vector(30.0, 30.0)
    shapleyCompare((1 to 8).map(_ => trainingData1).flatten, Vector[Any](1.0, 1.0), expected1)

    // Second example from Lundberg paper (https://arxiv.org/pdf/1802.03888.pdf)
    val trainingData2 = Seq(
      (Vector(1.0, 1.0), 90.0),
      (Vector(1.0, 0.0), 10.0),
      (Vector(0.0, 1.0), 0.0),
      (Vector(0.0, 0.0), 0.0)
    )
    val expected2 = Vector(35.0, 30.0)
    shapleyCompare((1 to 8).map(_ => trainingData2).flatten, Vector[Any](1.0, 1.0), expected2)

    // Example with two splits on one feature
    // Worked out with pen-and-paper from Lundberg Equation 2.
    val trainingData3 = Seq(
      (Vector(1.0, 1.0), 100.0),
      (Vector(1.0, 0.0), 80.0),
      (Vector(1.0, 0.2), 70.0),
      (Vector(0.0, 1.0), 0.0),
      (Vector(0.0, 0.2), 0.0),
      (Vector(0.0, 0.0), 0.0)
    )
    val expected3 = Vector(45.8333333333333, 12.5)
    shapleyCompare((1 to 8).map(_ => trainingData3).flatten, Vector[Any](1.0, 1.0), expected3)

    // Example with 5 features, to exercise all the factorials in Lundberg equation 2.
    // Referenced against the shap package on a sklearn decision tree.
    val trainingData4 = Seq(
      (Vector(0.0, 0.0, 0.0, 0.0, 0.0), 1.0),
      (Vector(1.0, 0.0, 0.0, 0.0, 0.0), 2.0),
      (Vector(0.0, 1.0, 0.0, 0.0, 0.0), 4.0),
      (Vector(0.0, 0.0, 1.0, 0.0, 0.0), 8.0),
      (Vector(0.0, 0.0, 0.0, 1.0, 0.0), 16.0),
      (Vector(0.0, 0.0, 0.0, 0.0, 1.0), 32.0)
    )
    val expected4 = Vector(0.0333333333333333, 0.2, 0.8666666666666667, 3.533333333333333, 16.866666666666667)
    shapleyCompare((1 to 8).map { _ => trainingData4 }.flatten, Vector.fill[Any](5)(1.0), expected4)
  }
}

object RandomForestTest {
  def main(argv: Array[String]): Unit = {
    new RandomForestTest()
      .testClassificationForest()
    new RandomForestTest()
      .testClassificationForestUnbiased()
    new RandomForestTest()
      .testShapley()
  }
}
