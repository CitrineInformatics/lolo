package io.citrine.lolo.linear

/**
  * Created by maxhutch on 11/15/16.
  */
class GuessTheMeanLearner {

  def train(trainingData: Seq[(Vector[Any], Vector[Double])]): GuessTheMeanModel = {
    val mean = trainingData.map(_._2).reduce((v1, v2) => v1.zip(v2).map(p => p._1 + p._2))

    new GuessTheMeanModel(mean)
  }
}

class GuessTheMeanModel(mean: Vector[Double]) {
  def transform(inputs: Seq[Vector[Any]]): Seq[Vector[Double]] = {
    Seq.fill(inputs.size)(mean)
  }
}
