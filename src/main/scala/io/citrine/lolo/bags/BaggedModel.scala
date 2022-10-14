package io.citrine.lolo.bags

import breeze.linalg.DenseMatrix
import io.citrine.lolo.Model

import scala.collection.parallel.immutable.ParSeq

/** A model holding a parallel sequence of models and the sample counts used to train them. */
trait BaggedModel[+T] extends Model[T] {

  def models: ParSeq[Model[T]]

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult[T]

  override def shapley(input: Vector[Any], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
    val ensembleShapley = models.map(model => model.shapley(input, omitFeatures))
    if (ensembleShapley.head.isEmpty) {
      None
    } else {
      assert(ensembleShapley.forall(x => x.isDefined))

      def sumReducer(a: Option[DenseMatrix[Double]], b: Option[DenseMatrix[Double]]): Option[DenseMatrix[Double]] = {
        Some(a.get + b.get)
      }

      val scale = 1.0 / ensembleShapley.length
      Some(scale * ensembleShapley.reduce(sumReducer).get)
    }
  }
}

class BaggedRegressionModel(
    val models: ParSeq[Model[Double]],
    Nib: Vector[Vector[Int]],
    rescale: Double = 1.0,
    disableBootstrap: Boolean = false,
    biasModel: Option[Model[Double]] = None
) extends BaggedModel[Double] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedRegressionResult = {
    assert(inputs.forall(_.size == inputs.head.size))

    val bias = biasModel.map(_.transform(inputs).getExpected())
    val ensemblePredictions = models.map(model => model.transform(inputs)).seq

    if (inputs.size == 1) {
      // In the special case of a single prediction on a real value, emit an optimized prediction class
      SinglePredictionBaggedResult(
        ensemblePredictions,
        Nib,
        bias.map(_.head),
        rescale,
        disableBootstrap
      )
    } else {
      MultiPredictionBaggedResult(
        ensemblePredictions,
        Nib,
        bias,
        rescale,
        disableBootstrap
      )
    }
  }
}

class BaggedClassificationModel(val models: ParSeq[Model[Any]], Nib: Vector[Vector[Int]]) extends BaggedModel[Any] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedClassificationResult = {
    assert(inputs.forall(_.size == inputs.head.size))
    val ensemblePredictions = models.map(model => model.transform(inputs)).seq
    BaggedClassificationResult(ensemblePredictions)
  }
}
