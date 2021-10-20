package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{Model, PredictionResult}

trait MultiModel extends Model[PredictionResult[Seq[Any]]] {
  def numLabels: Int

  def getModels: Seq[Model[PredictionResult[Any]]]
}

class MixedTree(models: Seq[Model[PredictionResult[Any]]]) extends MultiModel {
  override def numLabels: Int = models.length

  override def getModels: Seq[Model[PredictionResult[Any]]] = models

  override def transform(inputs: Seq[Vector[Any]]): MixedTreeResult = new MixedTreeResult(
    models.map { model =>
      model.transform(inputs).getExpected()
    }.transpose
  )
}

class MixedTreeResult(predictions: Seq[Seq[Any]]) extends PredictionResult[Seq[Any]] {
  override def getExpected(): Seq[Seq[Any]] = predictions
}