package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{Model, PredictionResult}

trait MultiModel extends Model[PredictionResult[Seq[Any]]] {
  val numLabels: Int

  def getRealLabels: Seq[Boolean]

  def getModels: Seq[Model[PredictionResult[Any]]]
}

class MixedTree(models: Seq[Model[PredictionResult[Any]]], realLabels: Seq[Boolean]) extends MultiModel {
  override val numLabels: Int = models.length

  override def getRealLabels: Seq[Boolean] = realLabels

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