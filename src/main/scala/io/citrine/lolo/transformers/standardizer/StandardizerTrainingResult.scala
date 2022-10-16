package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.TrainingResult

/** Training result wrapping the base learner's training result next to the transformations. */
trait StandardizerTrainingResult[+T] extends TrainingResult[T] {

  def baseTrainingResult: TrainingResult[T]

  override def getModel(): StandardizerModel[T]

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()
}

/**
  * Training result wrapping the input & output transformations for a regression learner.
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param outputTrans transformation (rescale, offset) of output label
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
case class RegressionStandardizerTrainingResult(
    baseTrainingResult: TrainingResult[Double],
    outputTrans: Standardization,
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerTrainingResult[Double] {

  override def getModel(): RegressionStandardizerModel =
    RegressionStandardizerModel(baseTrainingResult.getModel(), outputTrans, inputTrans)

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Double, Double)]] = {
    baseTrainingResult.getPredictedVsActual().map { pva =>
      pva.map {
        case (inputs, pred, actual) =>
          (Standardization.invertMulti(inputs, inputTrans), outputTrans.invert(pred), outputTrans.invert(actual))
      }
    }
  }
}

/**
  * Training result wrapping the input transformations for a classification learner.
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
case class ClassificationStandardizerTrainingResult[T](
    baseTrainingResult: TrainingResult[T],
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerTrainingResult[T] {

  override def getModel(): ClassificationStandardizerModel[T] =
    ClassificationStandardizerModel(baseTrainingResult.getModel(), inputTrans)

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], T, T)]] = {
    baseTrainingResult.getPredictedVsActual().map { pva =>
      pva.map {
        case (inputs, pred, actual) => (Standardization.invertMulti(inputs, inputTrans), pred, actual)
      }
    }
  }
}
