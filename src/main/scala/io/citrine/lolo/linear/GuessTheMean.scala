package io.citrine.lolo.linear

import io.citrine.random.Random
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult, TrainingRow}

case class GuessTheMeanLearner() extends Learner[Double] {

  override def train(trainingData: Seq[TrainingRow[Double]], rng: Random): GuessTheMeanTrainingResult = {
    val totalWeight = trainingData.map(_.weight).sum
    val mean = trainingData.map { case TrainingRow(_, label, weight) => label * weight }.sum / totalWeight
    GuessTheMeanTrainingResult(GuessTheMeanModel(mean))
  }
}

case class GuessTheMeanTrainingResult(model: GuessTheMeanModel) extends TrainingResult[Double]

case class GuessTheMeanModel(mean: Double) extends Model[Double] {
  def transform(inputs: Seq[Vector[Any]]): GuessTheMeanResult = {
    GuessTheMeanResult(Seq.fill(inputs.size)(mean))
  }
}

case class GuessTheMeanResult(expected: Seq[Double]) extends PredictionResult[Double]
