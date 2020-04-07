package io.citrine.lolo.learners

import breeze.stats.distributions.Beta
import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

import scala.util.Random

@Test
class ExtraRandomTreesTest {
  val rng = new Random(92345L)

  /**
    * Test that the regression does the same thing as the regression bagger
    */
  @Test
  def testRegression(): Unit = {
    rng.setSeed(982534L)
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, seed = rng.nextLong()),
      inputBins = Seq((0, 8))
    )

    Seq(true, false).foreach { randomlyRotateFeatures =>
      val RFMeta = ExtraRandomTrees(randomlyRotateFeatures = randomlyRotateFeatures, rng = rng)
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

  /**
    * Test that the regression does the same thing as the classification bagger
    */
  @Test
  def testClassification(): Unit = {
    rng.setSeed(25723478834L)

    val nTrain = 256
    val nTest = nTrain
    val nBins = 8
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(nTrain, 5, noise = 0.0, function = Friedman.friedmanSilverman, seed = rng.nextLong()),
      responseBins = Some(nBins)
    )

    /* Generate small perturbations from the training data */
    val testData = trainingData.map { r =>
      (r._1.map { _.asInstanceOf[Double] + 0.05 * rng.nextGaussian() }, r._2)
    }.take(nTest)

    Seq(true, false).foreach { randomlyRotateFeatures =>
      Seq(true, false).foreach { disableBootstrap =>
        Seq(1, 3).foreach { minLeafInstances =>
          rng.setSeed(238834L)

          val RFMeta = ExtraRandomTrees(numTrees = trainingData.size * 4, randomlyRotateFeatures = randomlyRotateFeatures, disableBootstrap = disableBootstrap, rng = rng, minLeafInstances = minLeafInstances)
            .train(trainingData)
          val RF = RFMeta.getModel()

          /* Inspect the results on the training set */
          val trueRateTrainingSet = trainingData.map(_._2).zip(RF.transform(trainingData.map(_._1)).getExpected()).count { case (a, p) => a == p }.toDouble / nTrain
          if (minLeafInstances == 1 && disableBootstrap) {
            assert(trueRateTrainingSet == 1.0)
          } else {
            assert(trueRateTrainingSet > 0.9)
          }

          /* Inspect the results on the perturbed test set */
          val results = RF.transform(testData.map(_._1))
          val means = results.getExpected()
          val trueRateTestSet = testData.map(_._2).zip(means).count { case (a, p) => a == p }.toDouble / nTest
          assert(trueRateTestSet > 0.5)

          /* Check that class probabilities are reasonable */
          val uncertainty = results.getUncertainty()
          assert(uncertainty.isDefined)
          assert(testData.map(_._2).zip(uncertainty.get).forall { case (a, probs) =>
            val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
            val maxProb = classProbabilities(a)
            maxProb > 1.0/(2*nBins) && maxProb < 1.0 && Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6
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
    rng.setSeed(257834L)
    val numTrials = 20
    val (winsSuffixed, winsPrefixed): (Int,Int) = (0 until numTrials).map { _ =>
      val nTrain = 64
      val nTest = 16
      val (mainTrainingData, testData) = {
        val allData = TestUtils.binTrainingData(
          TestUtils.generateTrainingData(nTrain + nTest, 5, noise = 0.1, function = Friedman.friedmanSilverman, seed = rng.nextLong()),
          responseBins = Some(2)
        )
        (allData.take(nTrain), allData.takeRight(nTest))
      }
      val dupeLabel = "DUPE"
      val trainingDataSuffixed = mainTrainingData ++ Seq(
        (mainTrainingData.head._1, dupeLabel)
      )
      val trainingDataPrefixed = Seq(
        (mainTrainingData.head._1, dupeLabel)
      ) ++ mainTrainingData

      val RFSuffixed = ExtraRandomTrees(numTrees = trainingDataSuffixed.size * 2, rng = rng).train(trainingDataSuffixed)
      val RFPrefixed = ExtraRandomTrees(numTrees = trainingDataPrefixed.size * 2, rng = rng).train(trainingDataPrefixed)
      val predictedSuffixed = RFSuffixed.getModel().transform(testData.map(_._1))
      val predictedPrefixed = RFPrefixed.getModel().transform(testData.map(_._1))
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
    * Make sure that we can draw training weights consistently even when the training size is small
    */
  @Test
  def testWeightsWithSmallData(): Unit = {
    rng.setSeed(7834L)
    val trainingData = TestUtils.generateTrainingData(8, 1, seed = rng.nextLong())
    // the number of trees is the number of times we generate weights
    // so this has the effect of creating lots of different sets of weights
    val learner = ExtraRandomTrees(numTrees = 16384, rng = rng)
    // the test is that this training doesn't throw an exception
    learner.train(trainingData).getModel()
  }

}

object ExtraRandomTreesTest {
  def main(argv: Array[String]): Unit = {
    new ExtraRandomTreesTest()
      .testClassification()
    new ExtraRandomTreesTest()
      .testClassificationUnbiased()
  }
}
