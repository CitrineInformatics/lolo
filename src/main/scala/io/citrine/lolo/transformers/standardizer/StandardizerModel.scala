package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.api.Model

trait StandardizerModel[+T] extends Model[T] {

  /** The base model trained on standardized data. */
  def baseModel: Model[T]

  /** Standardize the inputs and apply the base model. */
  override def transform(inputs: Seq[Vector[Any]]): StandardizerPrediction[T]
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
