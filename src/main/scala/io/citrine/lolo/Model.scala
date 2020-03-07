package io.citrine.lolo

import breeze.linalg.DenseVector

/**
  * Created by maxhutch on 11/14/16.
  */
trait Model[+T <: PredictionResult[Any]] extends Serializable {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a PredictionResult which includes, at least, the expected outputs
    */
  def transform(inputs: Seq[Vector[Any]]): T

  /**
    * Compute Shapley feature attributions for a given input in this node's subtree
    *
    * @param input for which to compute feature attributions.
    * @return array of vector-valued attributions for each feature
    *         One DenseVector[Double] per feature, each of length equal to the output dimension.
    */
  def shapley(input: Vector[Any]): Option[Vector[DenseVector[Double]]] = None
}
