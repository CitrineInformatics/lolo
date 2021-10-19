package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{Model, PredictionResult}

class MixedTree(models: Seq[Model[PredictionResult[Any]]]) extends Model[MixedTreeResult] {
  override def transform(inputs: Seq[Vector[Any]]): MixedTreeResult = new MixedTreeResult(
    models.map { model =>
      model.transform(inputs).getExpected()
    }.transpose
  )
}

class MixedTreeResult(predictions: Seq[Seq[Any]]) extends PredictionResult[Seq[Any]] {
  override def getExpected(): Seq[Seq[Any]] = predictions
}