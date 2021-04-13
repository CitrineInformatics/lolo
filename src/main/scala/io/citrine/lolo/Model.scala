package io.citrine.lolo

import breeze.linalg.DenseMatrix

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
    * @param omitFeatures feature indices to omit in computing Shapley values
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  def shapley(
               input: Vector[Any],
               omitFeatures: Set[Int] = Set()
             ): Option[DenseMatrix[Double]] = None

}
