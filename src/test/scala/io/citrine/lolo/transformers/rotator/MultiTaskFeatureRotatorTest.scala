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
    val data = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .data

    // Generate multi-task training data
    val (inputs, doubleLabel) = data.unzip
    val catLabel = inputs.map(x => Friedman.friedmanGrosseSilverman(x.asInstanceOf[Seq[Double]]) > 15.0)

    // Sparsify the categorical labels
    val sparseCatLabel = catLabel.map(x => if (rng.nextBoolean()) x else null)
    val labels = Vector(doubleLabel, sparseCatLabel).transpose

    // Train and evaluate rotated models on original and rotated features
    val baseLearner = MultiTaskTreeLearner()
    val rotatedLearner = MultiTaskFeatureRotator(MultiTaskTreeLearner())

    val baseTrainingResult = baseLearner.train(inputs.zip(labels), weights = None, rng = rng)
    val baseDoubleModel = baseTrainingResult.getModels().head
    val baseCatModel = baseTrainingResult.getModels().last
    val rotatedTrainingResult = rotatedLearner.train(inputs.zip(labels), weights = None, rng = rng)
    val rotatedDoubleModel = rotatedTrainingResult.getModels().head.asInstanceOf[RotatedFeatureModel[Double]]
    val rotatedCatModel = rotatedTrainingResult.getModels().last.asInstanceOf[RotatedFeatureModel[Boolean]]

    // Check double labels are the same
    val baseDoubleResult = baseDoubleModel.transform(inputs).getExpected()
    val rotatedDoubleResult = rotatedDoubleModel.transform(inputs).getExpected()
    baseDoubleResult.zip(rotatedDoubleResult).foreach {
      case (br: Double, rr: Double) =>
        assert(Math.abs(br - rr) < 1e-9, "Predicted double label not the same in rotated learner.")
    }

    // Check categorical labels are close
    val baseCatResult = baseCatModel.transform(inputs).getExpected()
    val rotatedCatResult = rotatedCatModel.transform(inputs).getExpected()
    val baseF1 = ClassificationMetrics.f1scores(baseCatResult, catLabel)
    val rotatedF1 = ClassificationMetrics.f1scores(rotatedCatResult, catLabel)
    // rotatedF1 appears to come in substantially lower than baseF1; this is just a rough sanity check / smoketest.
    assert(Math.abs(baseF1 - rotatedF1) < 0.15, s"baseF1 ${baseF1} and rotatedF1 ${rotatedF1} are too dissimilar.")

    // Check that rotation features and rotation matrices are the same for both tasks
    val U = rotatedDoubleModel.trans
    val rotatedFeatures = rotatedDoubleModel.rotatedFeatures
    assert(U == rotatedCatModel.trans)
    assert(rotatedFeatures == rotatedCatModel.rotatedFeatures)

    val rotatedInputs = FeatureRotator.applyRotation(data.map(_._1), rotatedFeatures, U)
    assert(
      rotatedDoubleModel.baseModel.transform(rotatedInputs).getExpected().zip(rotatedDoubleResult).count {
        case (a: Double, b: Double) => Math.abs(a - b) > 1e-9
      } == 0,
      "Rotated data passed to base model should map to the same predictions."
    )
    assert(
      rotatedDoubleModel.transform(rotatedInputs).getExpected().zip(baseDoubleResult).count {
        case (a: Double, b: Double) => Math.abs(a - b) > 1e-9
      } > 0,
      "Rotated data passed to rotated model should map to different predictions."
    )

    // Check that categorical labels change when we feed in different data.
    assert(
      rotatedCatModel
        .transform(rotatedInputs)
        .getExpected()
        .zip(baseCatModel.transform(rotatedInputs).getExpected())
        .count {
          case (a: Boolean, b: Boolean) => a != b
        } > 0,
      s"Categorical labels should substantially change when we feed in different inputs."
    )
    // Check that categorical labels are the same as feeding rotated data into base learner.
    assert(
      rotatedCatModel
        .transform(inputs)
        .getExpected()
        .zip(rotatedCatModel.baseModel.transform(rotatedInputs).getExpected())
        .count {
          case (a: Boolean, b: Boolean) => a != b
        } == 0,
      s"Categorical labels should be identical when we feed rotated inputs into base learner."
    )
  }
}
