package io.citrine.lolo.trees.classification

import io.citrine.lolo.{Model, PredictionResult}
import io.citrine.lolo.trees.{ModelNode, TreeMeta}

/**
  * The leaves just predict the most common value
  */
class ClassificationLeaf(model: Model[PredictionResult[Char]], depth: Int) extends ModelNode[PredictionResult[Char]] {
  override def transform(input: Vector[AnyVal]): (PredictionResult[Char], TreeMeta) = {
    (model.transform(Seq(input)), TreeMeta(depth))
  }
}
