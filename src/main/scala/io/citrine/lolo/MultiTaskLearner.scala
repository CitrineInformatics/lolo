package io.citrine.lolo

import io.citrine.lolo.trees.multitask.MultiModelTrainingResult

/**
  * A learner that trains on multiple labels, outputting one model per label.
  */
trait MultiTaskLearner extends Serializable {

  /**
    * Train models
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return a sequence of training results, one for each label
    */
  def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[TrainingResult]
}

/**
  * A learner that trains on multiple labels, outputting a single model that makes predictions for all labels.
  */
trait MultiTaskCombinedLearner extends Serializable {

  /**
    * Train a model
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return A training result that encompasses model(s) for all labels.
    */
  def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): MultiModelTrainingResult
}