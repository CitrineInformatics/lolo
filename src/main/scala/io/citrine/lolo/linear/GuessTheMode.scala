package io.citrine.lolo.linear

import io.citrine.random.Random
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult, TrainingRow}

case class GuessTheModeLearner[T]() extends Learner[T] {

  override def train(trainingData: Seq[TrainingRow[T]], rng: Random): TrainingResult[T] = {
    val mode = rng
      .shuffle { trainingData.groupBy(_.label).view.mapValues(_.map(_.weight).sum).toSeq }
      .maxBy(_._2)
      ._1
    GuessTheModeTrainingResult(GuessTheModeModel(mode))
  }
}

case class GuessTheModeTrainingResult[T](model: GuessTheModeModel[T]) extends TrainingResult[T] {
  override def getModel(): Model[T] = model
}

case class GuessTheModeModel[T](mean: T) extends Model[T] {

  def transform(inputs: Seq[Vector[Any]]): GuessTheModeResult[T] = {
    GuessTheModeResult(Seq.fill(inputs.size)(mean))
  }
}

case class GuessTheModeResult[T](result: Seq[T]) extends PredictionResult[T] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = result
}
