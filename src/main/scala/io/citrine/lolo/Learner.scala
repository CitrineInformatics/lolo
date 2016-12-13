package io.citrine.lolo

import io.citrine.lolo.results.TrainingResult

/**
  * Created by maxhutch on 11/14/16.
  */
abstract class Learner extends Serializable {

  def setHypers(moreHypers: Map[String, Any]): this.type = {
    hypers = hypers ++ moreHypers
    this
  }
  var hypers: Map[String, Any]

  def setHyper(name: String, value: Any): this.type = {
    setHypers(Map(name -> value))
  }

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
