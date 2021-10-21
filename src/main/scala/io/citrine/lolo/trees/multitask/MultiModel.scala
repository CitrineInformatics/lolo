package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{Model, PredictionResult, TrainingResult}

/** A model that predicts a sequence of values, corresponding to multiple labels. */
trait MultiModel extends Model[PredictionResult[Seq[Any]]] {
  /** The number of labels. Every prediction must have this length. */
  val numLabels: Int

  /** A boolean sequence indicating which labels are real-valued. Its length must be equal to `numLabels`. */
  def getRealLabels: Seq[Boolean]

  /** Individual models corresponding to each label */
  def getModels: Seq[Model[PredictionResult[Any]]]
}

/** A container that holds a multi-model prediction. */
class MultiModelResult(predictions: Seq[Seq[Any]]) extends PredictionResult[Seq[Any]] {
  override def getExpected(): Seq[Seq[Any]] = predictions
}

// TODO: fix this import, since I moved it from one file to another
trait MultiModelTrainingResult extends TrainingResult {
  override def getModel(): MultiModel

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Seq[Any], Seq[Any])]] = None
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

  override def transform(inputs: Seq[Vector[Any]]): MultiModelResult = new MultiModelResult(
    models.map { model =>
      model.transform(inputs).getExpected()
    }.transpose
  )
}
