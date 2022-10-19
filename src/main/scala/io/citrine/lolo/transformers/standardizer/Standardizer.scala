package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.Learner
import io.citrine.random.Random

trait Standardizer[T] extends Learner[T] {

  /** The base learner trained on the standardized data. */
  def baseLearner: Learner[T]

  /**
    * Create affine transformations for continuous features/labels; pass data through to the base learner.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]] = None,
      rng: Random = Random()
  ): StandardizerTrainingResult[T]
}

/** A standardizer on both inputs & outputs for regression tasks. */
case class RegressionStandardizer(baseLearner: Learner[Double]) extends Standardizer[Double] {

  override def train(
      trainingData: Seq[(Vector[Any], Double)],
      weights: Option[Seq[Double]],
      rng: Random
  ): RegressionStandardizerTrainingResult = {
    val inputTrans = Standardization.buildMulti(trainingData.map(_._1))
    val outputTrans = Standardization.build(trainingData.map(_._2))

    val (inputs, labels) = trainingData.unzip
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardLabels = labels.map(outputTrans.apply)
    val standardTrainingData = standardInputs.zip(standardLabels)

    val baseTrainingResult = baseLearner.train(standardTrainingData, weights, rng)
    RegressionStandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

/** A standardizer on features for classification tasks. */
case class ClassificationStandardizer[T](baseLearner: Learner[T]) extends Standardizer[T] {

  override def train(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]],
      rng: Random
  ): ClassificationStandardizerTrainingResult[T] = {
    val inputTrans = Standardization.buildMulti(trainingData.map(_._1))

    val (inputs, labels) = trainingData.unzip
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardTrainingData = standardInputs.zip(labels)

    val baseTrainingResult = baseLearner.train(standardTrainingData, weights, rng)
    ClassificationStandardizerTrainingResult(baseTrainingResult, inputTrans)
  }
}
