package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TrainingRow}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.{ClassificationTree, ClassificationTreeLearner}
import io.citrine.lolo.trees.regression.RegressionTree
import io.citrine.random.Random
import org.junit.Test
import org.scalatest.Assertions._

@Test
class MultiTaskTreeTest extends SeedRandomMixIn {

  // Set up some data.
  val realRows: Seq[TrainingRow[Double]] =
    DataGenerator.generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
  val (inputs: Seq[Vector[Double]], realLabel: Seq[Double]) = realRows.map(row => (row.inputs, row.label)).unzip
  val catLabel: Seq[Boolean] = realRows.map(_.label > realLabel.max / 2.0)
  val labels: Seq[Vector[Any]] = Vector(realLabel, catLabel).transpose
  val multiRows: Seq[TrainingRow[Vector[Any]]] = realRows.zip(labels).map { case (row, label) => row.withLabel(label) }

  /**
    * Test that the tree memorizes the data when it is complete
    */
  @Test
  def testTwoLabels(): Unit = {
    val learner = MultiTaskTreeLearner()
    val models = learner.train(multiRows, rng = rng).models
    assert(models.size == 2)
    assert(models.head.isInstanceOf[RegressionTree])
    assert(models.last.isInstanceOf[ClassificationTree])
    val realResults = models.head.transform(inputs).expected.asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(inputs).expected.asInstanceOf[Seq[Boolean]]

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
    val sparseRows = TrainingRow.build(inputs.zip(sparseLabels))

    val learner = MultiTaskTreeLearner()
    val models = learner.train(sparseRows).models
    val realResults = models.head.transform(inputs).expected.asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(inputs).expected.asInstanceOf[Seq[Boolean]]

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

    val sparseCatRows = TrainingRow.build(inputs.zip(sparseCat))
    val sparseMultiRows = TrainingRow.build(inputs.zip(sparseLabels)).filterNot(_.label == null)

    val learner = MultiTaskTreeLearner()
    val models = learner.train(sparseMultiRows, rng = rng).models
    val catResults = models.last.transform(inputs).expected.asInstanceOf[Seq[Boolean]]

    val reference = ClassificationTreeLearner()
      .train(sparseCatRows, rng = rng)
      .model
      .transform(inputs)
      .expected

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
    val trainRng = Random(seed)
    val combinedModelRng = Random(seed)
    val learner = MultiTaskTreeLearner()
    val combinedModelLearner = MultiTaskTreeLearner()
    val models = learner.train(TrainingRow.build(inputs.zip(labels)), rng = trainRng).models
    val combinedModel =
      combinedModelLearner.train(TrainingRow.build(inputs.zip(labels)), rng = combinedModelRng).model

    // Generate new inputs to test equality on.
    val testInputs = DataGenerator
      .generate(32, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .data
      .map(_.inputs)
    val realResults = models.head.transform(testInputs).expected.asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(testInputs).expected.asInstanceOf[Seq[Boolean]]
    val allResults = combinedModel.transform(testInputs).expected.asInstanceOf[Seq[Seq[Any]]]
    assert(Seq(realResults, catResults).transpose == allResults)
  }

  /** Test that feature importance correctly identifies the 1st feature as the most important for Friedman-Silverman. */
  @Test
  def testFeatureImportance(): Unit = {
    val multiTaskLearner = MultiTaskTreeLearner()
    val multiTaskTrainingResult = multiTaskLearner.train(TrainingRow.build(inputs.zip(labels)), rng = rng)
    val importances = multiTaskTrainingResult.featureImportance.get
    assert(importances(1) == importances.max)
  }
}
