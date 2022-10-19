package io.citrine.lolo

import io.citrine.random.Random

/**
  * Base trait for a supervised learner that produces a model.
  *
  * @tparam T the label type of the data the learner is trained on
  */
trait Learner[T] extends Serializable {

  /**
    * Train a model on the provided training data.
    *
    * @param trainingData to train on
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  def train(trainingData: Seq[TrainingRow[T]], rng: Random = Random()): TrainingResult[T]
}

/** A learner that trains on multiple labels, outputting a single model that makes predictions for all labels. */
trait MultiTaskLearner extends Learner[Vector[Any]] {

  def train(trainingData: Seq[TrainingRow[Vector[Any]]], rng: Random = Random()): MultiTaskTrainingResult
}
