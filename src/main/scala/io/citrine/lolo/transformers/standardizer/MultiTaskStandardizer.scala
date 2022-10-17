package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.{Model, MultiTaskLearner, MultiTaskModel, MultiTaskTrainingResult, ParallelModels}
import io.citrine.random.Random

/**
  * Standardize the training data to zero mean and unit variance before feeding it into a multi-task learner.
  *
  * @param baseLearner base multi-task learner to train on standardized data.
  */
case class MultiTaskStandardizer(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  override def train(
      trainingData: Seq[(Vector[Any], Vector[Any])],
      weights: Option[Seq[Double]],
      rng: Random
  ): MultiTaskStandardizerTrainingResult = {
    val (inputs, labels) = trainingData.unzip

    val inputTrans = Standardization.buildMulti(inputs)
    val outputTrans = Standardization.buildMulti(labels)

    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardLabels = labels.map { label => Standardization.applyMulti(label, outputTrans) }

    val baseTrainingResult = baseLearner.train(standardInputs.zip(standardLabels), weights, rng)
    new MultiTaskStandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param outputTrans sequence of optional transformation (rescale, offset) of output labels
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
class MultiTaskStandardizerTrainingResult(
    baseTrainingResult: MultiTaskTrainingResult,
    outputTrans: Seq[Option[Standardization]],
    inputTrans: Seq[Option[Standardization]]
) extends MultiTaskTrainingResult {

  override def getModel(): MultiTaskModel =
    new ParallelModels(getModels(), baseTrainingResult.getModel().getRealLabels)

  override def getModels(): Seq[StandardizerModel[Any]] = {
    val realLabels = getModel().getRealLabels
    baseTrainingResult.getModels().zipWithIndex.map {
      case (model, idx) =>
        if (realLabels(idx)) {
          RegressionStandardizerModel(model.asInstanceOf[Model[Double]], outputTrans(idx).get, inputTrans)
        } else {
          ClassificationStandardizerModel(model, inputTrans)
        }
    }
  }

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] = {
    baseTrainingResult.getPredictedVsActual().map { pva =>
      pva.map {
        case (inputs, predOpt, actualOpt) =>
          val invertedInputs = Standardization.invertMulti(inputs, inputTrans)
          val invertedPred = Standardization.invertMultiOption(predOpt, outputTrans)
          val invertedActual = Standardization.invertMultiOption(actualOpt, outputTrans)
          (invertedInputs, invertedPred, invertedActual)
      }
    }
  }
}
