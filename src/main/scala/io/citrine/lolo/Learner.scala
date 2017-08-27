package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
trait Learner extends Serializable {

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
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return training result containing a model
    */
  def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): TrainingResult

  /**
    * Train a model with weights
    *
    * @param trainingData with weights in the form (features, label, weight)
    * @return training result containing a model
    */
  def train(trainingData: Seq[(Vector[Any], Any, Double)]): TrainingResult = {
    train(trainingData.map(r => (r._1, r._2)), Some(trainingData.map(_._3)))
  }

}
