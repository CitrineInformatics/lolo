package io.citrine.lolo

/**
  * Container for prediction results; must include expected values
  * Created by maxhutch on 11/29/16.
  */
trait PredictionResult {

  /**
    * Get the expected values for this prediction
    * @return expected value of each prediction
    */
  def getExpected(): Seq[Any]
}

/**
  * This result includes uncertainty information
  */
trait withUncertainty {
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
trait withScores {
  /**
    * Get the training row scores for each prediction
    * @return training row scores of each prediction
    */
  def getScores(): Seq[Seq[Double]]
}

/**
  * This result includes gradient and/or sensitivity information
  */
trait withGradient {
  /**
    * Get the gradient or sensitivity of each prediction
    * @return a vector of doubles for each prediction
    */
  def getGradient(): Seq[Vector[Double]]
}
