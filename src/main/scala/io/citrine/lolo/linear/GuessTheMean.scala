package io.citrine.lolo.linear

import io.citrine.lolo.results.{PredictionResult, TrainingResult}
import io.citrine.lolo.{Learner, Model}

/**
  * Created by maxhutch on 11/15/16.
  */
class GuessTheMeanLearner extends Learner {

  override var hypers: Map[String, Any] = Map()

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
  /**
    * Get the hyperparameters used to train this model
    *
    * @return hypers set for model
    */
  override def getHypers(): Map[String, Any] = Map.empty[String, Any]

  override def getModel(): Model[GuessTheMeanResult] = model
}

@SerialVersionUID(1000L)
class GuessTheMeanModel(mean: Any) extends Model[GuessTheMeanResult] {

  def transform(inputs: Seq[Vector[Any]]): GuessTheMeanResult = {
    new GuessTheMeanResult(Seq.fill(inputs.size)(mean))
  }
}

class GuessTheMeanResult(result: Seq[Any]) extends PredictionResult[Any] {
  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = result
}
