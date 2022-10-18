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
    val inputs = trainingData.map(_.inputs)
    val featuresToRotate = FeatureRotator.getDoubleFeatures(inputs.head)
    val trans = FeatureRotator.getRandomRotation(inputs.head.length, rng)

    val rotatedFeatures = FeatureRotator.applyRotation(inputs, featuresToRotate, trans)
    val rotatedTrainingData = trainingData.zip(rotatedFeatures).map {
      case (row, inputs) => row.copy(inputs = inputs)
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

  override def getModel(): MultiTaskModel = new ParallelModels(getModels(), baseTrainingResult.getModel().getRealLabels)

  override def getModels(): Seq[Model[Any]] =
    baseTrainingResult.getModels().map { model =>
      RotatedFeatureModel(model, rotatedFeatures, trans)
    }

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] = {
    baseTrainingResult.getPredictedVsActual().map { pva =>
      pva.map {
        case (inputs, predicted, actual) =>
          (FeatureRotator.applyOneRotation(inputs, rotatedFeatures, trans), predicted, actual)
      }
    }
  }
}
