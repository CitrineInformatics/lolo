package io.citrine.lolo.results

/**
  * Container for prediction results; must include expected values
  * Created by maxhutch on 11/29/16.
  */
trait PredictionResult[+T] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  def getExpected(): Seq[T]
}

case class MultiResult[T](values: Seq[T]) extends PredictionResult[T] {
  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = values

  def append(other: this.type): MultiResult[T] = MultiResult(values ++ other.values)
}

/**
  * This result includes uncertainty information
  */
trait hasUncertainty {
  /**
    * Get the uncertainty of the prediction
    *
    * For example, in regression this is sqrt(bias^2 + variance)
    * @return uncertainty of each prediction
    */
  def getUncertainty(): Seq[Any]
}

/**
  * This result includes training row score information
  */
trait hasTrainingScores {
  /**
    * Get the training row scores for each prediction
    *
    * @return training row scores of each prediction
    */
  def getScores(): Seq[Seq[Double]]
}

/**
  * This result includes gradient and/or sensitivity information
  */
trait hasGradient {
  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  def getGradient(): Seq[Vector[Double]]
}
