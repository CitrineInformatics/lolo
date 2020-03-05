package io.citrine.lolo.learners

import breeze.stats.distributions.Beta
import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

/**
  * Created by maxhutch on 1/9/17.
  */
@Test
class RandomForestTest {

  /**
    * Test that the regression forest does the same thing as the regression bagger
    */
  @Test
  def testRegressionForest(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta = RandomForest(randomlyRotateFeatures = randomlyRotateFeatures)
        .train(trainingData)
      val RF = RFMeta.getModel()

      assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

      val results = RF.transform(trainingData.map(_._1))
      val means = results.getExpected()
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

  /**
    * Test that the regression forest does the same thing as the classification bagger
    */
  @Test
  def testClassificationForest(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta = new RandomForest(numTrees = trainingData.size * 2, randomlyRotateFeatures = randomlyRotateFeatures)
        .train(trainingData)
      val RF = RFMeta.getModel()

      /* Inspect the results */
      val results = RF.transform(trainingData.map(_._1))
      val means = results.getExpected()
      assert(trainingData.map(_._2).zip(means).forall { case (a, p) => a == p })

      val uncertainty = results.getUncertainty()
      assert(uncertainty.isDefined)
      assert(trainingData.map(_._2).zip(uncertainty.get).forall { case (a, probs) =>
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
    val numTrials = 30
    val (winsSuffixed, winsPrefixed): (Int,Int) = (0 until numTrials).map {case i: Int =>
      val mainTrainingData = TestUtils.binTrainingData(
        TestUtils.generateTrainingData(128, 5, noise = 0.1, function = Friedman.friedmanSilverman, seed = i),
        responseBins = Some(2)
      )
      val dupeLabel = "DUPE"
      val trainingDataSuffixed = mainTrainingData ++ Seq(
        (mainTrainingData.head._1, dupeLabel)
      )
      val trainingDataPrefixed = Seq(
        (mainTrainingData.head._1, dupeLabel)
      ) ++ mainTrainingData

      val RFSuffixed = RandomForest(numTrees = trainingDataSuffixed.size * 2).train(trainingDataSuffixed)
      val RFPrefixed = RandomForest(numTrees = trainingDataPrefixed.size * 2).train(trainingDataPrefixed)
      val predictedSuffixed = RFSuffixed.getModel().transform(mainTrainingData.map(_._1))
      val predictedPrefixed = RFPrefixed.getModel().transform(mainTrainingData.map(_._1))
      val extraLabelCountSuffixed = predictedSuffixed.getExpected().count { case p: String => p == dupeLabel }
      val extraLabelCountPrefixed = predictedPrefixed.getExpected().count { case p: String => p == dupeLabel }

      if (extraLabelCountSuffixed > extraLabelCountPrefixed) {
        (1,0)
      } else if (extraLabelCountSuffixed < extraLabelCountPrefixed) {
        (0,1)
      } else {
        (0,0)
      }
    }.asInstanceOf[Seq[(Int,Int)]].reduce{(a: (Int,Int),b: (Int,Int))=>(a._1 + b._1, a._2 + b._2)}

    // Posterior beta distribution with Jeffreys prior.
    val d = new Beta(winsSuffixed + 0.5, winsPrefixed + 0.5)
    val l = d.inverseCdf(2e-6)
    val r = d.inverseCdf(1-2e-6)
    val tol = 1e-2
    assert(l < 0.5 - tol, f"Bias detected toward prefixed duplicate rows: rate ${d.mean}%.3f (1e-6 CI ${l}%.3f - ${r}%.3f) should be close to 0.5")
    assert(r > 0.5 + tol, f"Bias detected toward suffixed duplicate rows: rate ${d.mean}%.3f (1e-6 CI ${l}%.3f - ${r}%.3f) should be close to 0.5")
  }

  /**
    * Randomized splits should do really well on linear signals when there are lots of trees.  Test that they
    * outperform mid-point splits
    */
  @Test
  def testRandomizedSplitLocations(): Unit = {
    // Generate a linear signal in one dimension: 2 * x
    val trainingData: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(32, 1, function = {x =>
      x.head * 2.0
    })

    // Create a consistent set of parameters
    val baseForest = RandomForest(numTrees = 16384, useJackknife = false)

    // Turn off split randomization and compute the loss (out-of-bag error)
    val lossWithoutRandomization: Double = baseForest.copy(randomizePivotLocation = false)
      .train(trainingData)
      .getLoss().get

    // Turn on split randomization and compute the loss (out-of-bag error)
    val lossWithRandomization: Double = baseForest.copy(randomizePivotLocation = true)
      .train(trainingData)
      .getLoss().get

    assert(lossWithRandomization < lossWithoutRandomization)
  }

  /**
    * Make sure that we can draw training weights consistently even when the training size is small
    */
  @Test
  def testWeightsWithSmallData(): Unit = {
    val trainingData = TestUtils.generateTrainingData(8, 1)
    // the number of trees is the number of times we generate weights
    // so this has the effect of creating lots of different sets of weights
    val learner = RandomForest(numTrees = 16384)
    // the test is that this training doesn't throw an exception
    learner.train(trainingData).getModel()
  }

}

object RandomForestTest {
  def main(argv: Array[String]): Unit = {
    new RandomForestTest()
      .testClassificationForest()
    new RandomForestTest()
      .testClassificationForestUnbiased()
  }
}
