package io.citrine.lolo.api

import breeze.linalg.DenseMatrix

trait Model[+T] extends Serializable {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a PredictionResult which includes, at least, the expected outputs
    */
  def transform(inputs: Seq[Vector[Any]]): PredictionResult[T]

  /**
    * Compute Shapley feature attributions for a given input in this node's subtree
    *
    * @param input for which to compute feature attributions.
    * @param omitFeatures feature indices to omit in computing Shapley values
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  def shapley(input: Vector[Any], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = None
}

/** A model that predicts a sequence of values, corresponding to multiple labels. */
trait MultiTaskModel extends Model[Vector[Any]] {

  /** The number of labels. Every prediction must have this length. */
  def numLabels: Int

  /** A boolean sequence indicating which labels are real-valued. Its length must be equal to `numLabels`. */
  def realLabels: Seq[Boolean]

  /** Individual models corresponding to each label */
  def models: Seq[Model[Any]]

  override def transform(inputs: Seq[Vector[Any]]): MultiTaskModelPredictionResult

}

/**
  * A set of parallel models, one for each label.
  *
  * @param models     sequence of models, one for each label
  * @param realLabels boolean sequence indicating which labels are real-valued
  */
case class ParallelModels(models: Seq[Model[Any]], realLabels: Seq[Boolean]) extends MultiTaskModel {
  override def numLabels: Int = models.length

  override def transform(inputs: Seq[Vector[Any]]): ParallelModelsPredictionResult = {
    val predictions = models.map(_.transform(inputs).expected).toVector.transpose
    ParallelModelsPredictionResult(predictions)
  }
}
