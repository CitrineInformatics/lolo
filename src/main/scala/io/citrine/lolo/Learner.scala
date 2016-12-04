package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
abstract class Learner extends Serializable {

  /**
    * Train a model
    * @param trainingData to train on
    * @param weights for the training rows, if applicable
    * @return a model
    */
  def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): Model

}
