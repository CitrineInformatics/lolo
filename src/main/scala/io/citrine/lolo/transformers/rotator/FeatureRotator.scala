package io.citrine.lolo.transformers.rotator

import io.citrine.lolo._
import io.citrine.random.Random
import breeze.linalg.{DenseMatrix, DenseVector, diag, qr}
import breeze.linalg.qr.QR
import breeze.numerics.signum
import breeze.stats.distributions.Gaussian
import io.citrine.lolo.api.{Learner, Model, PredictionResult, TrainingResult, TrainingRow}
import io.citrine.lolo.stats.StatsUtils.breezeRandBasis

/**
  * Rotate the training data before passing along to a base learner.
  *
  * This may be useful for improving randomization in random forests,
  * especially when using random feature selection without bagging.
  */
case class FeatureRotator[T](baseLearner: Learner[T]) extends Learner[T] {

  /**
    * Create linear transformations for continuous features and labels & pass data through to learner
    *
    * @param trainingData to train on
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(trainingData: Seq[TrainingRow[T]], rng: Random): RotatedFeatureTrainingResult[T] = {
    val featuresToRotate = FeatureRotator.getDoubleFeatures(trainingData.head.inputs)
    val trans = FeatureRotator.getRandomRotation(featuresToRotate.length, rng)

    val baseInputs = trainingData.map(_.inputs)
    val rotatedInputs = FeatureRotator.applyRotation(baseInputs, featuresToRotate, trans)
    val rotatedTrainingData = trainingData.zip(rotatedInputs).map {
      case (row, rotated) => row.withInputs(rotated)
    }

    val baseTrainingResult = baseLearner.train(rotatedTrainingData, rng)
    RotatedFeatureTrainingResult(baseTrainingResult, featuresToRotate, trans)
  }
}

/**
  * Training result bundling the base learner's training result with the list of rotated features and transformation.
  *
  * @param baseTrainingResult training result to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  */
case class RotatedFeatureTrainingResult[T](
    baseTrainingResult: TrainingResult[T],
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends TrainingResult[T] {

  override def model: Model[T] = RotatedFeatureModel(baseTrainingResult.model, rotatedFeatures, trans)

  override def loss: Option[Double] = baseTrainingResult.loss

  override def predictedVsActual: Option[Seq[(Vector[Any], T, T)]] = {
    baseTrainingResult.predictedVsActual.map { x =>
      x.map {
        case (v, e, a) => (FeatureRotator.applyOneRotation(v, rotatedFeatures, trans), e, a)
      }
    }
  }
}

/**
  * Model bundling the base learner's model with the list of rotated features and the transformation.
  *
  * @param baseModel model to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  * @tparam T label type
  */
case class RotatedFeatureModel[T](
    baseModel: Model[T],
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends Model[T] {

  /**
    * Rotate the inputs and then apply the base model.
    *
    * @param inputs to apply the model to
    * @return a RotatedFeaturePrediction which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): RotatedFeaturePrediction[T] = {
    val rotatedInputs = FeatureRotator.applyRotation(inputs, rotatedFeatures, trans)
    RotatedFeaturePrediction(baseModel.transform(rotatedInputs), rotatedFeatures, trans)
  }
}

/**
  * Prediction bundling the base learner's prediction with the list of rotated features and the transformation
  *
  * @param baseResult predictions made on rotated inputs
  * @param rotatedFeatures indices of features to rotate
  * @param trans rotation matrix to apply to features
  * @tparam T label type
  */
case class RotatedFeaturePrediction[T](
    baseResult: PredictionResult[T],
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends PredictionResult[T] {

  /**
    * Get the expected values for this prediction by delegating to the baseResult.
    *
    * @return expected value of each prediction
    */
  override def expected: Seq[T] = baseResult.expected

  /**
    * Get the uncertainty of the prediction by delegating to baseResult.
    *
    * @return uncertainty of each prediction
    */
  override def uncertainty(observational: Boolean): Option[Seq[Any]] = baseResult.uncertainty(observational)

  override def gradient: Option[Seq[Vector[Double]]] = {
    baseResult.gradient.map { g =>
      FeatureRotator.applyRotation(g, rotatedFeatures, trans.t).asInstanceOf[Seq[Vector[Double]]]
    }
  }
}

/**
  * Utilities to compute and apply rotations.
  */
object FeatureRotator {

  /**
    * Draw a random unitary matrix from the uniform (Haar) measure.
    *
    * @param dimension for which to get a rotator
    * @param rng random number generator, for reproducibility
    * @return unitary matrix
    */
  def getRandomRotation(dimension: Int, rng: Random = Random()): DenseMatrix[Double] = {
    val randBasis = breezeRandBasis(rng)
    val gaussian = Gaussian(0, 1)(randBasis)
    val X = DenseMatrix.rand(dimension, dimension, gaussian)
    val QR(_Q, _R) = qr(X)
    val d = signum(diag(_R))
    val detV = d.reduce((a, b) => a * b)
    detV * diag(d) * _Q.toDenseMatrix
  }

  /**
    * Get list of feature indices that make sense to rotate.
    *
    * @param rep representative vector of features
    * @return list of feature indices that are doubles
    */
  def getDoubleFeatures(rep: Vector[Any]): IndexedSeq[Int] = {
    rep.indices.filter(i => rep(i).isInstanceOf[Double])
  }

  /**
    * Apply rotation to a vector.
    *
    * @param input vector to rotate
    * @param featuresToRotate vector of feature indices included in rotation
    * @param trans linear transformation matrix to apply
    * @return rotated vectors
    */
  def applyOneRotation(
      input: Vector[Any],
      featuresToRotate: IndexedSeq[Int],
      trans: DenseMatrix[Double]
  ): Vector[Any] = {
    val out = input.toArray
    val rotated: DenseVector[Double] =
      trans * DenseVector(featuresToRotate.map(i => input(i)).asInstanceOf[Seq[Double]].toArray)
    featuresToRotate.indices.foreach { i =>
      out(featuresToRotate(i)) = rotated(i)
    }
    out.toVector
  }

  /**
    * Apply rotation to a sequence of vectors.
    *
    * @param input sequence of vectors to rotate
    * @param featuresToRotate vector of feature indices included in rotation
    * @param trans linear transformation matrix to apply
    * @return sequence of rotated vectors
    */
  def applyRotation(
      input: Seq[Vector[Any]],
      featuresToRotate: IndexedSeq[Int],
      trans: DenseMatrix[Double]
  ): Seq[Vector[Any]] = {
    input.map { x => applyOneRotation(x, featuresToRotate, trans) }
  }
}
