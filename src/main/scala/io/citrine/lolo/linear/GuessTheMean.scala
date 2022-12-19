package io.citrine.lolo.linear

import io.citrine.lolo.api.{Learner, Model, PredictionResult, TrainingResult, TrainingRow}
import io.citrine.random.Random

/** Learner that computes the (weighted) mean value over real-valued labels. */
case class GuessTheMeanLearner() extends Learner[Double] {

  override def train(trainingData: Seq[TrainingRow[Double]], rng: Random): GuessTheMeanTrainingResult[Double] = {
    val trainingLabels = trainingData.map(_.label)
    val allEqual = trainingLabels.forall(_ == trainingLabels.head)
    val mean = if (allEqual) {
      trainingLabels.head
    } else {
      val totalWeight = trainingData.map(_.weight).sum
      trainingData.map { case TrainingRow(_, label, weight) => label * weight }.sum / totalWeight
    }
    GuessTheMeanTrainingResult(GuessTheMeanModel(mean))
  }
}

/** Learner that computes the (weighted) mode value over labels of type `T`. */
case class GuessTheModeLearner[T]() extends Learner[T] {

  override def train(trainingData: Seq[TrainingRow[T]], rng: Random): GuessTheMeanTrainingResult[T] = {
    val mode = rng
      .shuffle { trainingData.groupBy(_.label).view.mapValues(_.map(_.weight).sum).toSeq }
      .maxBy(_._2)
      ._1
    GuessTheMeanTrainingResult(GuessTheMeanModel(mode))
  }
}

case class GuessTheMeanTrainingResult[+T](model: GuessTheMeanModel[T]) extends TrainingResult[T]

case class GuessTheMeanModel[+T](value: T) extends Model[T] {
  def transform(inputs: Seq[Vector[Any]]): GuessTheMeanPrediction[T] = {
    GuessTheMeanPrediction(Seq.fill(inputs.size)(value))
  }
}

case class GuessTheMeanPrediction[+T](expected: Seq[T]) extends PredictionResult[T]
