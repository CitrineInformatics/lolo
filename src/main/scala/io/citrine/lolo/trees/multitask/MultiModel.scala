package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{Model, PredictionResult, TrainingResult}

/** Container for predictions made on multiple labels simultaneously. */
trait MultiModelPredictionResult extends PredictionResult[Seq[Any]] {

  override def getUncertainty(observational: Boolean = true): Option[Seq[Seq[Any]]] = None

  // TODO: should this interface include an `observational` option? Is there a difference to the correlation coefficient?
  /**
    * Get the correlation coefficients between the predictions made on two labels.
    * Correlation coefficient is bounded between -1 and 1.
    * If either index is out of bounds or does not correspond to a real-valued label, then this method must reutrn None.
    *
    * @param  i index of the first label
    * @param  j index of the second label
    * @return optional sequence of correlation coefficients between specified labels for each prediction
    */
  def getUncertaintyCorrelation(i: Int, j: Int): Option[Seq[Double]] = None

}

/** A model that predicts a sequence of values, corresponding to multiple labels. */
trait MultiModel extends Model[MultiModelPredictionResult] {
  /** The number of labels. Every prediction must have this length. */
  val numLabels: Int

  /** A boolean sequence indicating which labels are real-valued. Its length must be equal to `numLabels`. */
  def getRealLabels: Seq[Boolean]

  /** Individual models corresponding to each label */
  def getModels: Seq[Model[PredictionResult[Any]]]
}

trait MultiModelTrainingResult extends TrainingResult {
  override def getModel(): MultiModel

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Seq[Any], Seq[Any])]] = None
}

/** A container that holds a multi-model prediction. */
class MultiModelDefinedResult(predictions: Seq[Seq[Any]]) extends MultiModelPredictionResult {
  override def getExpected(): Seq[Seq[Any]] = predictions
}

/**
  * A multi-model that contains a separate model for each label. As the class is used currently, these models
  * are trees with identical split structures. But that does not need to be the case.
  *
  * @param models     sequence of models, one for each label
  * @param realLabels boolean sequence indicating which labels are real-valued
  */
class ParallelModels(models: Seq[Model[PredictionResult[Any]]], realLabels: Seq[Boolean]) extends MultiModel {
  override val numLabels: Int = models.length

  override def getRealLabels: Seq[Boolean] = realLabels

  override def getModels: Seq[Model[PredictionResult[Any]]] = models

  override def transform(inputs: Seq[Vector[Any]]): MultiModelPredictionResult = new MultiModelDefinedResult(
    models.map { model =>
      model.transform(inputs).getExpected()
    }.transpose
  )
}
