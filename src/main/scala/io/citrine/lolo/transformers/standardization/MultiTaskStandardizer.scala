package io.citrine.lolo.transformers.standardization

import io.citrine.lolo.{Model, MultiTaskLearner, MultiTaskModel, MultiTaskTrainingResult, ParallelModels}
import io.citrine.lolo.transformers.{Standardization, Standardizer, StandardizerModel}
import io.citrine.random.Random

case class MultiTaskStandardizer(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  /**
    * Train a model
    *
    * @param trainingData  to train on
    * @param weights for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a sequence of training results, one for each label
    */
  override def train(
      trainingData: Seq[(Vector[Any], Vector[Any])],
      weights: Option[Seq[Double]],
      rng: Random
  ): MultiTaskStandardizerTrainingResult = {
    val (inputs, labels) = trainingData.unzip
    val labelsTransposed = labels.transpose.toVector
    val repOutput = labels.head
    val inputTrans = Standardization.getMultiStandardization(inputs)
    val outputTrans: Seq[Option[Standardization]] = repOutput.indices.map { i =>
      if (repOutput(i) != null && repOutput(i).isInstanceOf[Double]) {
        val labelSeq = labelsTransposed(i)
        Some(Standardization.getStandardization(labelSeq.asInstanceOf[Seq[Double]].filterNot(_.isNaN())))
      } else {
        None
      }
    }
    val standardInputs = Standardization.applyStandardization(inputs, inputTrans)
    val standardLabels = labelsTransposed
      .zip(outputTrans)
      .map {
        case (labelSeq, trans) =>
          Standardization.applyStandardization(labelSeq, trans).toVector
      }
      .transpose

    val baseTrainingResult = baseLearner.train(standardInputs.zip(standardLabels), weights, rng)
    new MultiTaskStandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param outputTrans optional sequence of transformation (rescale, offset) of output labels
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
class MultiTaskStandardizerTrainingResult(
    baseTrainingResult: MultiTaskTrainingResult,
    outputTrans: Seq[Option[Standardization]],
    inputTrans: Seq[Option[Standardization]]
) extends MultiTaskTrainingResult {

  override def getModel(): MultiTaskModel = new ParallelModels(getModels(), baseTrainingResult.getModel().getRealLabels)

  override def getModels(): Seq[Model[Any]] =
    baseTrainingResult.getModels().zipWithIndex.map {
      case (model, i) =>
        new StandardizerModel(model, outputTrans(i), inputTrans)
    }

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] = {
    baseTrainingResult.getPredictedVsActual() match {
      case None => None
      case Some(predictedVsActual) =>
        Some(
          Standardizer
            .invertStandardization(predictedVsActual.map(_._1), inputTrans)
            .lazyZip(Standardization.invertStandardizationOption(predictedVsActual.map(_._2), outputTrans))
            .lazyZip(Standardization.invertStandardizationOption(predictedVsActual.map(_._3), outputTrans))
            .toSeq
        )
    }
  }
}
