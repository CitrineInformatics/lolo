package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TrainingRow}
import org.junit.Test

class MultiTaskStandardizerTest extends SeedRandomMixIn {

  val data: Seq[TrainingRow[Double]] =
    DataGenerator.generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data

  /**
    * Test that multitask sandardization has the proper performance and invariants
    */
  @Test
  def testMultiTaskStandardizer(): Unit = {
    // Generate multi-task training data
    val (inputs, doubleLabel) = data.map(row => (row.inputs, row.label)).unzip
    val catLabel = inputs.map(x => Friedman.friedmanGrosseSilverman(x.asInstanceOf[Seq[Double]]) > 15.0)

    // Sparsify the categorical labels
    val sparseCatLabel = catLabel.map(x => if (rng.nextBoolean()) x else null)
    val labels = Vector(doubleLabel, sparseCatLabel).transpose
    val baseRows = data.zip(labels).map {
      case (row, label) => row.withLabel(label)
    }

    // Screw up the scale of the double labels
    val scale = 10000.0
    val rescaledLabel = doubleLabel.map(_ * scale)
    val rescaledLabels = rescaledLabel.zip(sparseCatLabel).map { case (r, c) => Vector(r, c) }
    val rescaledRows = data.zip(rescaledLabels).map {
      case (row, label) => row.withLabel(label)
    }

    // Train and evaluate standard models on original and rescaled labels
    val standardizer = MultiTaskStandardizer(MultiTaskTreeLearner())
    val baseRes = standardizer.train(baseRows).models.last.transform(inputs).expected
    val standardRes = standardizer.train(rescaledRows).models.last.transform(inputs).expected
    // Train and evaluate unstandardized model on rescaled labels

    // Compute metrics for each of the models
    val baseF1 = ClassificationMetrics.f1scores(baseRes, catLabel)
    val standardF1 = ClassificationMetrics.f1scores(standardRes, catLabel)

    // Assert some things
    assert(Math.abs(baseF1 - standardF1) < 1.0e-6, s"Expected training to be invariant the scale of the labels")
  }
}
