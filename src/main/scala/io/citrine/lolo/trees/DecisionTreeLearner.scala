package io.citrine.lolo.trees

import io.citrine.lolo.linear.{GuessTheMeanLearner, GuessTheMeanModel}
import io.citrine.lolo.{Learner, Model}

/**
  * Created by maxhutch on 11/15/16.
  */
class DecisionTreeLearner extends Learner {
  override def train(trainingData: Seq[(Vector[Any], Vector[Any])]): Model = {
    val gtm = new GuessTheMeanLearner().train(trainingData)
    new DecisionTreeModel(gtm)
  }
}

class DecisionTreeModel(gtm: GuessTheMeanModel) extends Model {
  override def transform(inputs: Seq[Vector[Any]]): Seq[Vector[Any]] = {
    gtm.transform(inputs)
  }
}