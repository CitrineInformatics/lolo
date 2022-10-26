package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.api.{Learner, TrainingRow}
import io.citrine.random.Random

trait Standardizer[T] extends Learner[T] {

  /** The base learner trained on the standardized data. */
  def baseLearner: Learner[T]

  /**
    * Create affine transformations for continuous features/labels; pass data through to the base learner.
    *
    * @param trainingData to train on
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(trainingData: Seq[TrainingRow[T]], rng: Random = Random()): StandardizerTrainingResult[T]
}

/** A standardizer on both inputs & outputs for regression tasks. */
case class RegressionStandardizer(baseLearner: Learner[Double]) extends Standardizer[Double] {

  override def train(
      trainingData: Seq[TrainingRow[Double]],
      rng: Random = Random()
  ): RegressionStandardizerTrainingResult = {
    val (inputs, labels, weights) = trainingData.map(_.asTuple).unzip3

    val inputTrans = Standardization.buildMulti(inputs)
    val outputTrans = Standardization.build(labels)

    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardLabels = labels.map(outputTrans.apply)
    val standardTrainingData = standardInputs.lazyZip(standardLabels).lazyZip(weights).map {
      case (i, l, w) => TrainingRow(i, l, w)
    }

    val baseTrainingResult = baseLearner.train(standardTrainingData, rng)
    RegressionStandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

/** A standardizer on features for classification tasks. */
case class ClassificationStandardizer[T](baseLearner: Learner[T]) extends Standardizer[T] {

  override def train(trainingData: Seq[TrainingRow[T]], rng: Random): ClassificationStandardizerTrainingResult[T] = {
    val inputs = trainingData.map(_.inputs)
    val inputTrans = Standardization.buildMulti(inputs)

    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardTrainingData = trainingData.zip(standardInputs).map {
      case (row, inputs) => row.withInputs(inputs)
    }

    val baseTrainingResult = baseLearner.train(standardTrainingData, rng)
    ClassificationStandardizerTrainingResult(baseTrainingResult, inputTrans)
  }
}
