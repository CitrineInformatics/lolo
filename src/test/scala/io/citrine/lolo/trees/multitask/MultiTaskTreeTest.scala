package io.citrine.lolo.trees.multitask

import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.{ClassificationTree, ClassificationTreeLearner}
import io.citrine.lolo.trees.regression.{RegressionTree, RegressionTreeLearner}
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class MultiTaskTreeTest {
  val rng = new Random(1012795L)

  /* Setup some data */
  val raw: Seq[(Vector[Double], Double)] =
    TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, seed = rng.nextLong())
  val (inputs: Seq[Vector[Double]], realLabel: Seq[Double]) = raw.unzip
  val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)
  val labels = Vector(realLabel, catLabel).transpose

  /**
    * Test that the tree memorizes the data when it is complete
    */
  @Test
  def testTwoLabels(): Unit = {
    val learner = MultiTaskTreeLearner()
    val models = learner.train(inputs.zip(labels)).getModels()
    assert(models.size == 2)
    assert(models.head.isInstanceOf[RegressionTree])
    assert(models.last.isInstanceOf[ClassificationTree])
    val realResults = models.head.transform(inputs).getExpected().asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]

    assert(realLabel.zip(realResults).forall(p => p._1 == p._2))
    assert(catLabel.zip(catResults).forall(p => p._1 == p._2))
  }

  /**
    * Test that the tree memorizes the data when it is sparse
    */
  @Test
  def testSparseExact(): Unit = {
    val sparseCat = catLabel.map(x =>
      if (rng.nextBoolean()) {
        null
      } else {
        x
      }
    )
    val sparseLabels = Vector(realLabel, sparseCat).transpose

    val learner = MultiTaskTreeLearner()
    val models = learner.train(inputs.zip(sparseLabels)).getModels()
    val realResults = models.head.transform(inputs).getExpected().asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]

    assert(realLabel.zip(realResults).forall(p => p._1 == p._2))
    assert(sparseCat.zip(catResults).forall(p => p._1 == null || p._1 == p._2))
  }

  /**
    * Test that the multitask tree on sparse boolean labels is better than single task
    */
  @Test
  def testSparseQuantitative(): Unit = {
    val sparseCat = catLabel.map(x =>
      if (rng.nextDouble() > 0.125) {
        null
      } else {
        x
      }
    )
    val sparseLabels = Vector(realLabel, sparseCat).transpose

    val learner = MultiTaskTreeLearner()
    val models = learner.train(inputs.zip(sparseLabels)).getModels()
    val catResults = models.last.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]

    val reference = ClassificationTreeLearner()
      .train(inputs.zip(sparseCat).filterNot(_._2 == null))
      .getModel()
      .transform(inputs)
      .getExpected()

    val singleF1 = ClassificationMetrics.f1scores(reference, catLabel)
    val multiF1 = ClassificationMetrics.f1scores(catResults, catLabel)
    assert(multiF1 < 1.0)
    assert(multiF1 > singleF1)
  }

  /** Test that the resulting predictions are independent of whether the trees are stored in one model or several models. */
  @Test
  def testSingleModelEquality(): Unit = {
    // Train twice with the same seed, first outputting two models and then outputting a combined model.
    val seed = 817235L
    val rng = new Random(seed)
    val combinedModelRng = new Random(seed)
    val learner = MultiTaskTreeLearner(rng = rng)
    val combinedModelLearner = MultiTaskTreeLearner(rng = combinedModelRng)
    val models = learner.train(inputs.zip(labels)).getModels()
    val combinedModel = combinedModelLearner.train(inputs.zip(labels)).getModel()

    // Generate new inputs to test equality on.
    val testInputs = TestUtils
      .generateTrainingData(32, 12, noise = 0.1, function = Friedman.friedmanSilverman, seed = rng.nextLong())
      .map(_._1)
    val realResults = models.head.transform(testInputs).getExpected().asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(testInputs).getExpected().asInstanceOf[Seq[Boolean]]
    val allResults = combinedModel.transform(testInputs).getExpected().asInstanceOf[Seq[Seq[Any]]]
    assert(Seq(realResults, catResults).transpose == allResults)
  }

  /** Test that feature importance correctly identifies the 1st feature as the most important for Friedman-Silverman. */
  @Test
  def testFeatureImportance(): Unit = {
    val seed = 253768L
    val multiTaskLearner = MultiTaskTreeLearner(rng = new Random(seed))
    val multiTaskTrainingResult = multiTaskLearner.train(inputs.zip(labels))
    val importances = multiTaskTrainingResult.getFeatureImportance().get
    assert(importances(1) == importances.max)
  }

}
