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
    val data = trainingData.map(_._2).zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0)))
    val mean = data.head._1 match {
      case x: Double => data.asInstanceOf[Seq[(Double, Double)]].map(p => p._1 * p._2).sum / data.map(_._2).sum
      case x: Any => data.groupBy(_._1).mapValues(_.map(_._2).sum).maxBy(_._2)._1
    }

    new GuessTheMeanTrainingResult(new GuessTheMeanModel(mean))
  }
}

@SerialVersionUID(999L)
class GuessTheMeanTrainingResult[T](model: GuessTheMeanModel[T]) extends TrainingResult {
  /**
    * Get the hyperparameters used to train this model
    *
    * @return hypers set for model
    */
  override def getHypers(): Map[String, Any] = Map.empty[String, Any]

  override def getModel(): Model[GuessTheMeanResult[T]] = model
}

@SerialVersionUID(1000L)
class GuessTheMeanModel[T](mean: T) extends Model[GuessTheMeanResult[T]] {

  def transform(inputs: Seq[Vector[Any]]): GuessTheMeanResult[T] = {
    new GuessTheMeanResult(Seq.fill(inputs.size)(mean))
  }
}

class GuessTheMeanResult[T](result: Seq[T]) extends PredictionResult[T] {
  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = result
}
