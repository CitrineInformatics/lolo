package io.citrine.lolo.bags

import breeze.linalg.DenseMatrix
import io.citrine.lolo.Model

import scala.collection.parallel.immutable.ParSeq

trait BaggedModel[+T] extends Model[T] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult[T]
}

/**
  * Container holding a parallel sequence of models and the sample counts used to train them
  *
  * @param models in this bagged model
  * @param Nib    training sample counts
  */
class BaggedRegressionModel(
    val models: ParSeq[Model[Double]],
    Nib: Vector[Vector[Int]],
    rescale: Double = 1.0,
    disableBootstrap: Boolean = false,
    biasModel: Option[Model[Double]] = None
) extends BaggedModel[Double] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult[Double] = {
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

  /**
    * Compute Shapley feature attributions for a given input
    *
    * @param input for which to compute feature attributions.
    * @param omitFeatures feature indices to omit in computing Shapley values
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
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

/**
  * Container holding a parallel sequence of models and the sample counts used to train them
  *
  * @param models in this bagged model
  * @param Nib    training sample counts
  */
class BaggedClassificationModel(
    val models: ParSeq[Model[Any]],
    Nib: Vector[Vector[Int]],
    disableBootstrap: Boolean = false
) extends BaggedModel[Any] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedClassificationResult = {
    assert(inputs.forall(_.size == inputs.head.size))
    val ensemblePredictions = models.map(model => model.transform(inputs)).seq
    BaggedClassificationResult(ensemblePredictions)
  }

  /**
    * Compute Shapley feature attributions for a given input
    *
    * @param input for which to compute feature attributions.
    * @param omitFeatures feature indices to omit in computing Shapley values
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
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
