package io.citrine.lolo.linear

import io.citrine.random.Random
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

case class GuessTheMeanLearner() extends Learner[Double] {

  override def train(
      trainingData: Seq[(Vector[Any], Double)],
      weights: Option[Seq[Double]],
      rng: Random
  ): GuessTheMeanTrainingResult = {
    val data = trainingData.map(_._2).zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0)))
    val mean = data.map(p => p._1 * p._2).sum / data.map(_._2).sum
    GuessTheMeanTrainingResult(new GuessTheMeanModel(mean))
  }
}

case class GuessTheMeanTrainingResult(model: GuessTheMeanModel) extends TrainingResult[Double] {
  override def getModel(): GuessTheMeanModel = model
}

class GuessTheMeanModel(mean: Double) extends Model[Double] {

  def transform(inputs: Seq[Vector[Any]]): GuessTheMeanResult = {
    new GuessTheMeanResult(Seq.fill(inputs.size)(mean))
  }
}

class GuessTheMeanResult(result: Seq[Double]) extends PredictionResult[Double] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = result
}
