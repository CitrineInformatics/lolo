package io.citrine.lolo.linear

import io.citrine.random.Random
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

case class GuessTheModeLearner[T]() extends Learner[T] {

  override def train(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]],
      rng: Random
  ): TrainingResult[T] = {
    val data = trainingData.map(_._2).zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0)))
    val mode = rng.shuffle(data.groupBy(_._1).view.mapValues(_.map(_._2).sum).toSeq).maxBy(_._2)._1
    GuessTheModeTrainingResult(new GuessTheModeModel(mode))
  }
}

case class GuessTheModeTrainingResult[T](model: GuessTheModeModel[T]) extends TrainingResult[T] {
  override def getModel(): Model[T] = model
}

class GuessTheModeModel[T](mean: T) extends Model[T] {

  def transform(inputs: Seq[Vector[Any]]): GuessTheModeResult[T] = {
    new GuessTheModeResult(Seq.fill(inputs.size)(mean))
  }
}

class GuessTheModeResult[T](result: Seq[T]) extends PredictionResult[T] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = result
}
