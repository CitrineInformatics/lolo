package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
trait MultiTaskLearner extends Serializable {

  /**
    * Get the hyperparameter map
    *
    * @return map of hyperparameters
    */
  def getHypers(): Map[String, Any]

  /**
    * Train a model
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return a sequence of training results, one for each label
    */
  def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[TrainingResult]
}
