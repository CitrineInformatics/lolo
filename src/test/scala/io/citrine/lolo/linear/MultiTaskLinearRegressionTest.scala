package io.citrine.lolo.linear

import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test
import org.scalatest.Assertions.assertThrows

class MultiTaskLinearRegressionTest extends SeedRandomMixIn {

  /** Test that multi-task linear is exceptional on mixed label types. */
  @Test
  def testNonRealLabel(): Unit = {
    val realRows = DataGenerator.generate(64, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
    val (inputs, realLabel) = realRows.map(row => (row.inputs, row.label)).unzip
    val catLabel = realRows.map(_.label > realLabel.max / 2.0)
    val labels = Vector(realLabel, catLabel).transpose
    val mixedRows = TrainingRow.build(inputs.zip(labels))

    val learner = MultiTaskLinearRegressionLearner()
    assertThrows[AssertionError](learner.train(mixedRows, rng = rng))
  }

  /** Test that training on multiple labels produces the expected models. */
  @Test
  def testMultipleLabels(): Unit = {
    val realRows = DataGenerator.generate(64, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
    val multiRealRows = realRows.map { row => row.mapLabel(y => Vector(y, 2.0 * y + 3.0)) }

    val learner = MultiTaskLinearRegressionLearner()
    val trainingResult = learner.train(multiRealRows, rng = rng)
    assert(trainingResult.models.length == 2)

    val model = trainingResult.model
    assert(model.realLabels.forall(identity))

    val testInputs =
      DataGenerator.generate(8, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data.map(_.inputs)
    val predictions = model.transform(testInputs)
    val expected = predictions.expected
    assert(expected.length == testInputs.length)
    assert(expected.forall(_.length == 2)) // 2 labels at each predicted point
    assert(predictions.uncertainty().isEmpty)
  }

  /** Test that we can train on non-overlapping labels. */
  @Test
  def testLabelSparsity(): Unit = {
    val realRows = DataGenerator.generate(64, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
    val label1Rows = realRows.map(_.mapLabel(y => Vector(y, Double.NaN)))
    val label2Rows = realRows.map(_.mapLabel(y => Vector(Double.NaN, 2.0 * y + 3.0)))
    val sparseRows = label1Rows ++ label2Rows

    val learner = MultiTaskLinearRegressionLearner()
    val model = learner.train(sparseRows, rng = rng).model

    val testInputs =
      DataGenerator.generate(16, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data.map(_.inputs)
    val expected = model.transform(testInputs).expected
    assert(expected.length == testInputs.length)
    assert(
      expected.forall { vec =>
        vec.forall {
          case x: Double if !x.isNaN => true
          case _                     => false
        }
      }
    )
  }

  /** Test that feature importance correctly identifies the most important features. */
  @Test
  def testFeatureImportance(): Unit = {
    val linearFunction: Seq[Double] => Double = x => 5.0 + 10.0 * x.head + 3.0 * x.last
    val realRows = DataGenerator.generate(64, 12, noise = 0.1, function = linearFunction, rng = rng).data
    val multiTaskRows = realRows.map(_.mapLabel(y => Vector(y, 2.0 * y + 3.0)))
    val multiTaskLearner = MultiTaskLinearRegressionLearner()
    val multiTaskTrainingResult = multiTaskLearner.train(multiTaskRows, rng = rng)
    val importance = multiTaskTrainingResult.featureImportance.get
    assert(importance.head == importance.max)
  }
}
