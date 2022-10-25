package io.citrine.lolo.transformers.rotator

import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import org.junit.Test

class MultiTaskFeatureRotatorTest extends SeedRandomMixIn {

  /**
    * Verify that rotated-input multi-task trees have expected properties
    */
  @Test
  def testMultiTaskRotator(): Unit = {
    val rawData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .data

    // Generate multi-task training data
    val inputs = rawData.map(_.inputs)
    val doubleLabel = rawData.map(_.label)
    val catLabel = inputs.map(x => Friedman.friedmanGrosseSilverman(x.asInstanceOf[Seq[Double]]) > 15.0)

    // Sparsify the categorical labels
    val sparseCatLabel = catLabel.map(x => if (rng.nextBoolean()) x else null)
    val labels = Vector(doubleLabel, sparseCatLabel).transpose
    val multiTaskData = rawData.zip(labels).map { case (row, label) => row.withLabel(label) }

    // Train and evaluate rotated models on original and rotated features
    val baseLearner = MultiTaskTreeLearner()
    val rotatedLearner = MultiTaskFeatureRotator(MultiTaskTreeLearner())

    val baseTrainingResult = baseLearner.train(multiTaskData, rng = rng)
    val baseDoubleModel = baseTrainingResult.models.head
    val baseCatModel = baseTrainingResult.models.last
    val rotatedTrainingResult = rotatedLearner.train(multiTaskData, rng = rng)
    val rotatedDoubleModel = rotatedTrainingResult.models.head.asInstanceOf[RotatedFeatureModel[Double]]
    val rotatedCatModel = rotatedTrainingResult.models.last.asInstanceOf[RotatedFeatureModel[Boolean]]

    // Check double labels are the same
    val baseDoubleResult = baseDoubleModel.transform(inputs).expected
    val rotatedDoubleResult = rotatedDoubleModel.transform(inputs).expected
    baseDoubleResult.zip(rotatedDoubleResult).foreach {
      case (br: Double, rr: Double) =>
        assert(Math.abs(br - rr) < 1e-9, "Predicted double label not the same in rotated learner.")
    }

    // Check categorical labels are close
    val baseCatResult = baseCatModel.transform(inputs).expected
    val rotatedCatResult = rotatedCatModel.transform(inputs).expected
    val baseF1 = ClassificationMetrics.f1scores(baseCatResult, catLabel)
    val rotatedF1 = ClassificationMetrics.f1scores(rotatedCatResult, catLabel)
    // rotatedF1 appears to come in substantially lower than baseF1; this is just a rough sanity check / smoketest.
    assert(Math.abs(baseF1 - rotatedF1) < 0.15, s"baseF1 ${baseF1} and rotatedF1 ${rotatedF1} are too dissimilar.")

    // Check that rotation features and rotation matrices are the same for both tasks
    val U = rotatedDoubleModel.trans
    val rotatedFeatures = rotatedDoubleModel.rotatedFeatures
    assert(U == rotatedCatModel.trans)
    assert(rotatedFeatures == rotatedCatModel.rotatedFeatures)

    val rotatedInputs = FeatureRotator.applyRotation(inputs, rotatedFeatures, U)
    assert(
      rotatedDoubleModel.baseModel.transform(rotatedInputs).expected.zip(rotatedDoubleResult).count {
        case (a: Double, b: Double) => Math.abs(a - b) > 1e-9
      } == 0,
      "Rotated data passed to base model should map to the same predictions."
    )
    assert(
      rotatedDoubleModel.transform(rotatedInputs).expected.zip(baseDoubleResult).count {
        case (a: Double, b: Double) => Math.abs(a - b) > 1e-9
      } > 0,
      "Rotated data passed to rotated model should map to different predictions."
    )

    // Check that categorical labels change when we feed in different data.
    assert(
      rotatedCatModel
        .transform(rotatedInputs)
        .expected
        .zip(baseCatModel.transform(rotatedInputs).expected)
        .count {
          case (a: Boolean, b: Boolean) => a != b
        } > 0,
      s"Categorical labels should substantially change when we feed in different inputs."
    )
    // Check that categorical labels are the same as feeding rotated data into base learner.
    assert(
      rotatedCatModel
        .transform(inputs)
        .expected
        .zip(rotatedCatModel.baseModel.transform(rotatedInputs).expected)
        .count {
          case (a: Boolean, b: Boolean) => a != b
        } == 0,
      s"Categorical labels should be identical when we feed rotated inputs into base learner."
    )
  }
}
