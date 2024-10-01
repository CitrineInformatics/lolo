package io.citrine.lolo.bags

import breeze.linalg.DenseMatrix
import io.citrine.lolo.api.Model

import scala.collection.parallel.CollectionConverters._

/** A model holding a parallel sequence of models and the sample counts used to train them. */
trait BaggedModel[+T] extends Model[T] {

  /** Models in the ensemble trained on subsets of the training data. */
  def ensembleModels: Seq[Model[T]]

  override def transform(inputs: Seq[Vector[Any]]): BaggedPrediction[T]

  override def shapley(input: Vector[Any], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
    val ensembleShapley: Seq[Option[DenseMatrix[Double]]] = ensembleModels.par.map(_.shapley(input, omitFeatures)).seq

    if (ensembleShapley.isEmpty || ensembleShapley.forall(_.isEmpty)) {
      None
    } else {
      val validMatrices: Seq[DenseMatrix[Double]] = ensembleShapley.flatten
      if (validMatrices.isEmpty) {
        None
      } else {
        val sumMatrix: DenseMatrix[Double] = validMatrices.reduce(_ + _)
        val scale = 1.0 / ensembleModels.length
        Some(scale * sumMatrix)
      }
    }
  }
}

case class BaggedRegressionModel(
    ensembleModels: Seq[Model[Double]],
    Nib: Vector[Vector[Int]],
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false,
    biasModel: Option[Model[Double]] = None
) extends BaggedModel[Double] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedRegressionPrediction = {
    assert(inputs.forall(_.size == inputs.head.size))

    val bias = biasModel.map(_.transform(inputs).expected)
    val ensemblePredictions = ensembleModels.par.map(model => model.transform(inputs)).seq

    if (inputs.size == 1) {
      // In the special case of a single prediction on a real value, emit an optimized prediction class
      SinglePointBaggedPrediction(
        ensemblePredictions,
        Nib,
        bias.map(_.head),
        rescaleRatio,
        disableBootstrap
      )
    } else {
      MultiPointBaggedPrediction(
        ensemblePredictions,
        Nib,
        bias,
        rescaleRatio,
        disableBootstrap
      )
    }
  }
}

case class BaggedClassificationModel[T](ensembleModels: Seq[Model[T]]) extends BaggedModel[T] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedClassificationPrediction[T] = {
    assert(inputs.forall(_.size == inputs.head.size))
    val ensemblePredictions = ensembleModels.par.map(model => model.transform(inputs)).seq
    BaggedClassificationPrediction(ensemblePredictions)
  }
}
