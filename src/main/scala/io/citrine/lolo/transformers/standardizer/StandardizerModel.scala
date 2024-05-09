package io.citrine.lolo.transformers.standardizer

import breeze.linalg.DenseMatrix
import io.citrine.lolo.api.Model

trait StandardizerModel[+T] extends Model[T] {

  /** The base model trained on standardized data. */
  def baseModel: Model[T]

  /** Standardize the inputs and apply the base model. */
  override def transform(inputs: Seq[Vector[Any]]): StandardizerPrediction[T]

  override def shapley(input: Vector[Any], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
      val baseModel_ = baseModel
      baseModel_.shapley(input, omitFeatures)
  }
}

case class RegressionStandardizerModel(
    baseModel: Model[Double],
    outputTrans: Standardization,
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerModel[Double] {

  override def transform(inputs: Seq[Vector[Any]]): RegressionStandardizerPrediction = {
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    RegressionStandardizerPrediction(baseModel.transform(standardInputs), outputTrans, inputTrans)
  }
}

case class ClassificationStandardizerModel[T](
    baseModel: Model[T],
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerModel[T] {

  override def transform(inputs: Seq[Vector[Any]]): ClassificationStandardizerPrediction[T] = {
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    ClassificationStandardizerPrediction(baseModel.transform(standardInputs), inputTrans)
  }
}
