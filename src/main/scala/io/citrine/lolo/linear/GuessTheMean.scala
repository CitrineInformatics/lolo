package io.citrine.lolo.linear

import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

/**
  * Created by maxhutch on 11/15/16.
  */
class GuessTheMeanLearner extends Learner {


  /**
    * Train a model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return training result containing a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): TrainingResult = {
    val data = trainingData.map(_._2.asInstanceOf[Double]).zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0)))
    val mean = data.map(p => p._1 * p._2).sum / data.map(_._2).sum

    new GuessTheMeanTrainingResult(new GuessTheMeanModel(mean))
  }
}

class GuessTheMeanTrainingResult(model: GuessTheMeanModel) extends TrainingResult {
  override def getModel(): Model = model
}

class GuessTheMeanModel(mean: Any) extends Model {

  def transform(inputs: Seq[Vector[Any]]): GuessTheMeanResult = {
    new GuessTheMeanResult(Seq.fill(inputs.size)(mean))
  }
}

class GuessTheMeanResult(result: Seq[Any]) extends PredictionResult {
  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = result
}
