package io.citrine.lolo.trees.classification

import io.citrine.lolo.{MultiResult, PredictionResult}
import io.citrine.lolo.trees.{ModelNode, TreeMeta}

/**
  * Created by maxhutch on 1/12/17.
  */
/**
  * The leaves just predict the most common value
  *
  * @param mode most common value
  */
class ClassificationLeaf(mode: Char, depth: Int) extends ModelNode[PredictionResult[Char]] {
  override def transform(input: Vector[AnyVal]): (PredictionResult[Char], TreeMeta) = (MultiResult(Seq(mode)), TreeMeta(depth))
}
