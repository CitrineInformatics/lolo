package io.citrine.lolo.linear

import io.citrine.lolo.{Learner, Model}

/**
  * Created by maxhutch on 11/15/16.
  */
class GuessTheMeanLearner extends Learner {

  override def train(trainingData: Seq[(Vector[Any], Vector[Any])]): GuessTheMeanModel = {
    val mean = trainingData.map(_._2).sum

    new GuessTheMeanModel(mean)
  }
}

class GuessTheMeanModel(mean: Vector[Any]) extends Model {
  override def transform(inputs: Seq[Vector[Any]]): Seq[Vector[Any]] = {
    Seq.fill(inputs.size)(mean)
  }
}
