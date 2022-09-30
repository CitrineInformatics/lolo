package io.citrine.lolo.transformers

import io.citrine.lolo._
import io.citrine.random.Random
import breeze.linalg.{diag, qr, DenseMatrix, DenseVector}
import breeze.linalg.qr.QR
import breeze.numerics.signum
import breeze.stats.distributions.Gaussian
import io.citrine.lolo.stats.StatsUtils.breezeRandBasis

/**
  * Rotate the training data before passing along to a base learner
  *
  * This may be useful for improving randomization in random forests,
  * especially when using random feature selection without bagging.
  */
case class FeatureRotator(baseLearner: Learner) extends Learner {

  /**
    * Create linear transformations for continuous features and labels & pass data through to learner
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): RotatedFeatureTrainingResult = {
    val featuresToRotate = FeatureRotator.getDoubleFeatures(trainingData.head._1)
    val trans = FeatureRotator.getRandomRotation(featuresToRotate.length, rng)

    val (inputs, labels) = trainingData.unzip
    val rotatedTrainingData = FeatureRotator.applyRotation(inputs, featuresToRotate, trans).zip(labels)
    val baseTrainingResult = baseLearner.train(rotatedTrainingData, weights, rng)

    RotatedFeatureTrainingResult(baseTrainingResult, featuresToRotate, trans)
  }
}

case class MultiTaskFeatureRotator(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  /**
    * Create linear transformations for continuous features and labels & pass data through to learner
    *
    * @param trainingData  to train on
    * @param weights for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a sequence of training results, one for each label
    */
  override def train(
      trainingData: Seq[(Vector[Any], Vector[Any])],
      weights: Option[Seq[Double]],
      rng: Random
  ): MultiTaskRotatedFeatureTrainingResult = {
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val featuresToRotate = FeatureRotator.getDoubleFeatures(inputs.head)
    val trans = FeatureRotator.getRandomRotation(inputs.head.length, rng)
    val rotatedFeatures = FeatureRotator.applyRotation(inputs, featuresToRotate, trans)
    val baseTrainingResult = baseLearner.train(rotatedFeatures.zip(labels), weights, rng)
    MultiTaskRotatedFeatureTrainingResult(baseTrainingResult, featuresToRotate, trans)
  }
}

/**
  * Training result bundling the base learner's training result with the list of rotated features and the transformation
  *
  * @param baseTrainingResult training result to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  */
case class RotatedFeatureTrainingResult(
    baseTrainingResult: TrainingResult,
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends TrainingResult {

  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[PredictionResult[Any]] = {
    RotatedFeatureModel(baseTrainingResult.getModel(), rotatedFeatures, trans)
  }

  override def getLoss(): Option[Double] = baseTrainingResult.getLoss()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Any, Any)]] = {
    baseTrainingResult.getPredictedVsActual().map { x =>
      x.map {
        case (v: Vector[Any], e: Any, a: Any) => (FeatureRotator.applyOneRotation(v, rotatedFeatures, trans), e, a)
      }
    }
  }
}

/**
  * Training result bundling the base learner's multitask training result with the list of rotated features and the transformation
  *
  * @param baseTrainingResult training result to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  */
case class MultiTaskRotatedFeatureTrainingResult(
    baseTrainingResult: MultiTaskTrainingResult,
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends MultiTaskTrainingResult {
  override def getModel(): MultiTaskModel = new ParallelModels(getModels(), baseTrainingResult.getModel().getRealLabels)

  override def getModels(): Seq[Model[PredictionResult[Any]]] =
    baseTrainingResult.getModels().map { model =>
      RotatedFeatureModel(model, rotatedFeatures, trans)
    }

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Seq[Option[Any]], Seq[Option[Any]])]] = {
    baseTrainingResult.getPredictedVsActual() match {
      case None => None
      case Some(predictedVsActual) =>
        Some(predictedVsActual.map {
          case (inputs: Vector[Any], predicted: Seq[Option[Any]], actual: Seq[Option[Any]]) =>
            (FeatureRotator.applyOneRotation(inputs, rotatedFeatures, trans), predicted, actual)
        })
    }
  }
}

/**
  * Model bundling the base learner's model with the list of rotated features and the transformation
  *
  * @param baseModel model to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  * @tparam T label type
  */
case class RotatedFeatureModel[T](
    baseModel: Model[PredictionResult[T]],
    rotatedFeatures: IndexedSeq[Int],
    trans: DenseMatrix[Double]
) extends Model[PredictionResult[T]] {

  /**
    * Transform the inputs and then apply the base model
    *
    * @param inputs to apply the model to
    * @return a RotatedFeaturePredictionResult which includes, at least, the expected outputs
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
    * Get the expected values for this prediction by delegating to baseResult
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = baseResult.getExpected()

  /**
    * Get the uncertainty of the prediction by delegating to baseResult
    *
    * @return uncertainty of each prediction
    */
  override def getUncertainty(observational: Boolean): Option[Seq[Any]] = baseResult.getUncertainty(observational)

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient().map { g =>
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
