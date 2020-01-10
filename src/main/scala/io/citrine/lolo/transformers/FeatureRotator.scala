package io.citrine.lolo.transformers

import io.citrine.lolo._
import breeze.linalg.{DenseMatrix, DenseVector, diag, qr}
import breeze.linalg.qr.QR
import breeze.numerics.signum
import breeze.stats.distributions.Gaussian

/**
  * Rotate the training data before passing along to a base learner
  *
  * This may be useful for improving randomization in random forests,
  * especially when using random feature selection without bagging.
  *
  * Created by gregor-robinson on 2020-01-02.
  */
case class FeatureRotator(baseLearner: Learner) extends Learner {

  /**
    * Create linear transformations for continuous features and labels & pass data through to learner
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return training result containing a model
    */
  override def train(
                     trainingData: Seq[(Vector[Any], Any)],
                     weights: Option[Seq[Double]]
                    ): RotatedFeatureTrainingResult = {
    val featuresToRotate = FeatureRotator.getDoubleFeatures(trainingData.head._1)
    val trans = FeatureRotator.getRandomRotation(featuresToRotate.length)

    val (inputs, labels) = trainingData.unzip
    val rotatedTrainingData = FeatureRotator.applyRotation(inputs, featuresToRotate, trans).zip(labels)
    val baseTrainingResult = baseLearner.train(rotatedTrainingData, weights)

    new RotatedFeatureTrainingResult(baseTrainingResult, featuresToRotate, trans)
  }
}

class MultiTaskFeatureRotator(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  /**
    * Create linear transformations for continuous features and labels & pass data through to learner
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return a sequence of training results, one for each label
    */
  override def train(
                     inputs: Seq[Vector[Any]],
                     labels: Seq[Seq[Any]],
                     weights: Option[Seq[Double]]
                    ): Seq[RotatedFeatureTrainingResult] = {
    val featuresToRotate = FeatureRotator.getDoubleFeatures(inputs.head)
    val trans = FeatureRotator.getRandomRotation(inputs.head.length)
    val rotatedFeatures = FeatureRotator.applyRotation(inputs, featuresToRotate, trans)
    val baseTrainingResult = baseLearner.train(rotatedFeatures, labels, weights)

    baseTrainingResult.map { case (base) =>
      new RotatedFeatureTrainingResult(base, featuresToRotate, trans)
    }
  }
}

/**
  * Training result bundling the base learner's training result with the list of rotated features and the transformation
  *
  * @param baseTrainingResult training result to which to delegate prediction on rotated features
  * @param rotatedFeatures indices of features to rotate
  * @param trans matrix to apply to features
  */
class RotatedFeatureTrainingResult(
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
    new RotatedFeatureModel(baseTrainingResult.getModel(), rotatedFeatures, trans)
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
class RotatedFeatureModel[T](
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
    val rotatedInputs = FeatureRotator.applyRotation(inputs, rotatedFeatures,  trans)
    new RotatedFeaturePrediction(baseModel.transform(rotatedInputs), rotatedFeatures, trans)
  }
}

/**
  * Prediction bundling the base learner's prediction with the list of rotated features and the transformation
  *
  * @param baseResult
  * @param rotatedFeatures
  * @param trans
  * @tparam T
  */
class RotatedFeaturePrediction[T](
                                  baseResult: PredictionResult[T],
                                  rotatedFeatures: IndexedSeq[Int],
                                  trans: DenseMatrix[Double]
                                 ) extends PredictionResult[T] {
  /**
    * Get the expected values for this prediction by delegating to baseResult
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = baseResult.getExpected().asInstanceOf[Seq[T]]

  /**
    * Get the uncertainty of the prediction by delegating to baseResult
    *
    * @return uncertainty of each prediction
    */
  override def getUncertainty(): Option[Seq[Any]] = baseResult.getUncertainty()

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
    * @return unitary matrix
    */
  def getRandomRotation(dimension: Int): DenseMatrix[Double] = {
    val X = DenseMatrix.rand(dimension, dimension, Gaussian(0, 1))
    val QR(_Q, _R) = qr(X)
    val d = signum(diag(_R))
    val detV = d.reduce((a,b) => a*b)
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
    val rotated: DenseVector[Double] = trans * DenseVector(featuresToRotate.map(i => input(i)).asInstanceOf[Seq[Double]].toArray)
    featuresToRotate.indices.foreach{ i =>
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
