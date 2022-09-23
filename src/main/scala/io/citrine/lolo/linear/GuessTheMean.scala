package io.citrine.lolo.linear

import io.citrine.random.Random
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

/**
  * Created by maxhutch on 11/15/16.
  */
case class GuessTheMeanLearner() extends Learner {

  /**
    * Train a model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): TrainingResult = {
    val data = trainingData.map(_._2).zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0)))
    val mean = data.head._1 match {
      case _: Double => data.asInstanceOf[Seq[(Double, Double)]].map(p => p._1 * p._2).sum / data.map(_._2).sum
      case _: Any    => rng.shuffle(data.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq).maxBy(_._2)._1
    }

    new GuessTheMeanTrainingResult(new GuessTheMeanModel(mean))
  }
}

class GuessTheMeanTrainingResult[T](model: GuessTheMeanModel[T]) extends TrainingResult {
  override def getModel(): Model[GuessTheMeanResult[T]] = model
}

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
