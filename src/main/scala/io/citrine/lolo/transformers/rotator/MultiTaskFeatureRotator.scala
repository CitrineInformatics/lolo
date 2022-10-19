package io.citrine.lolo.transformers.rotator

import breeze.linalg.DenseMatrix
import io.citrine.lolo.{Model, MultiTaskLearner, MultiTaskModel, MultiTaskTrainingResult, ParallelModels, TrainingRow}
import io.citrine.random.Random

case class MultiTaskFeatureRotator(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  /**
    * Create linear transformations for continuous features and labels & pass data through to learner
    *
    * @param trainingData  to train on
    * @param rng          random number generator for reproducibility
    * @return a sequence of training results, one for each label
    */
  override def train(
      trainingData: Seq[TrainingRow[Vector[Any]]],
      rng: Random
  ): MultiTaskRotatedFeatureTrainingResult = {
    val baseInputs = trainingData.map(_.inputs)
    val featuresToRotate = FeatureRotator.getDoubleFeatures(baseInputs.head)
    val trans = FeatureRotator.getRandomRotation(baseInputs.head.length, rng)

    val rotatedInputs = FeatureRotator.applyRotation(baseInputs, featuresToRotate, trans)
    val rotatedTrainingData = trainingData.zip(rotatedInputs).map {
      case (row, rotated) => row.withInputs(rotated)
    }

    val baseTrainingResult = baseLearner.train(rotatedTrainingData, rng)
    MultiTaskRotatedFeatureTrainingResult(baseTrainingResult, featuresToRotate, trans)
  }
}

/**
  * Training result bundling the base learner's multitask training result with the list of rotated features and the transformation
  *
  * @param baseTrainingResult training result to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  */
case class MultiTaskRotatedFeatureTrainingResult(
    baseTrainingResult: MultiTaskTrainingResult,
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends MultiTaskTrainingResult {

  override def model: MultiTaskModel = ParallelModels(models, baseTrainingResult.model.realLabels)

  override def models: Seq[Model[Any]] =
    baseTrainingResult.models.map { model => RotatedFeatureModel(model, rotatedFeatures, trans) }

  override def predictedVsActual: Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] = {
    baseTrainingResult.predictedVsActual.map { pva =>
      pva.map {
        case (inputs, predicted, actual) =>
          (FeatureRotator.applyOneRotation(inputs, rotatedFeatures, trans), predicted, actual)
      }
    }
  }
}
