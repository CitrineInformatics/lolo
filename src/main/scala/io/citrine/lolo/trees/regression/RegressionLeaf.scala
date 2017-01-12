package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.{ModelNode, TreeMeta}
import io.citrine.lolo.{Model, PredictionResult}

/**
  * Created by maxhutch on 1/12/17.
  */
class RegressionLeaf(model: Model[PredictionResult[Double]], depth: Int) extends ModelNode[PredictionResult[Double]] {
  override def transform(input: Vector[AnyVal]): (PredictionResult[Double], TreeMeta) = {
    (model.transform(Seq(input)), TreeMeta(depth))
  }
}
