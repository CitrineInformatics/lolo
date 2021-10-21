package io.citrine.lolo

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
trait MultiTaskCombinedLearner extends MultiTaskLearner {

  /**
    * Train a model
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return A training result that encompasses model(s) for all labels.
    */
  // TODO: explain that this must be a single-valued sequence (and see if there's a way to get rid of this constraint
  // Maybe add a numModels val to MultiTaskLearner, then fix it to 1 in MultiTaskCombinedLearner.
  // It won't prevent mismatch, but will allow for pre-emptive checks that numModels is as expected, and a clear error message.
  def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[MultiModelTrainingResult]
}