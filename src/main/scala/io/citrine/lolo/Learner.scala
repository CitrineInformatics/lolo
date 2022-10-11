package io.citrine.lolo

import io.citrine.random.Random

trait Learner[T] extends Serializable {

  /**
    * Train a model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  def train(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]] = None,
      rng: Random = Random()
  ): TrainingResult[T]

  def train(trainingData: Seq[(Vector[Any], T, Double)], rng: Random): TrainingResult[T] = {
    train(trainingData.map(r => (r._1, r._2)), Some(trainingData.map(_._3)), rng)
  }
}

/**
  * A learner that trains on multiple labels, outputting a single model that makes predictions for all labels.
  */
trait MultiTaskLearner extends Learner[Vector[Any]] {

  override def train(
      trainingData: Seq[(Vector[Any], Vector[Any])],
      weights: Option[Seq[Double]],
      rng: Random
  ): MultiTaskTrainingResult
}
