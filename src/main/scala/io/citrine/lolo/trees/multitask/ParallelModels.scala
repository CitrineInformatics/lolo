package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{Model, MultiTaskModel, MultiTaskModelPredictionResult, PredictionResult}

/**
  * A set of parallel models, one for each label. As the class is used currently, these models
  * are trees with identical split structures. But that does not need to be the case.
  *
  * @param models     sequence of models, one for each label
  * @param realLabels boolean sequence indicating which labels are real-valued
  */
class ParallelModels(models: Seq[Model[PredictionResult[Any]]], realLabels: Seq[Boolean]) extends MultiTaskModel {
  override val numLabels: Int = models.length

  override def getRealLabels: Seq[Boolean] = realLabels

  override def getModels: Seq[Model[PredictionResult[Any]]] = models

  override def transform(inputs: Seq[Vector[Any]]) = new MultiModelDefinedResult(
    models.map { model =>
      model.transform(inputs).getExpected()
    }.transpose
  )
}

/** A container that holds the predictions of several parallel models for multiple labels. */
class MultiModelDefinedResult(predictions: Seq[Seq[Any]]) extends MultiTaskModelPredictionResult {
  override def getExpected(): Seq[Seq[Any]] = predictions
}