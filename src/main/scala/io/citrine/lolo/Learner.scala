package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
abstract class Learner {

  def train(trainingData: Seq[(Vector[Any], Vector[Any])]): Model

}
