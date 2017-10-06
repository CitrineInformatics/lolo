package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
trait MultiTaskLearner extends Serializable {

  /**
    * Set block of hyperparameters
    * @param moreHypers hyperparameters to set
    * @return this learner
    */
  def setHypers(moreHypers: Map[String, Any]): this.type = {
    hypers = hypers ++ moreHypers
    this
  }
  var hypers: Map[String, Any] = Map()

  /**
    * Set a single hyperparameter
    * @param name of the hyperparameter
    * @param value of the hyperparameter
    * @return this learner
    */
  def setHyper(name: String, value: Any): this.type = {
    setHypers(Map(name -> value))
  }

  /**
    * Get the hyperparameter map
    * @return map of hyperparameters
    */
  def getHypers(): Map[String, Any] = hypers

  /**
    * Train a model
    *
    * @param inputs to train on
    * @param labels sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return a sequence of training results, one for each label
    */
  def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[TrainingResult]
}
