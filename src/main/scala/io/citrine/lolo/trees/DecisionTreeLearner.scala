package io.citrine.lolo.trees

import io.citrine.lolo.linear.{GuessTheMeanLearner, GuessTheMeanModel}

/**
  * Created by maxhutch on 11/15/16.
  */
class RegressionTreeLearner {
  def train(trainingData: Seq[(Vector[Any], Vector[Double])]): RegressionTreeModel = {
    val gtm = new GuessTheMeanLearner().train(trainingData)
    new RegressionTreeModel(gtm)
  }
}

class RegressionTreeModel(gtm: GuessTheMeanModel) {
  def transform(inputs: Seq[Vector[Any]]): Seq[Vector[Any]] = {
    gtm.transform(inputs)
  }
}

