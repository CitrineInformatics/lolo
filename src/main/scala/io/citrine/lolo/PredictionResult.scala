package io.citrine.lolo

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

  /**
    * Get the uncertainty of the prediction
    *
    * For example, in regression this is sqrt(bias^2 + variance)
    * @return uncertainty of each prediction
    */
  def getUncertainty(): Option[Seq[Any]] = None

  /**
    * Get the training row scores for each prediction
    *
    * @return training row scores of each prediction
    */
  def getImportanceScores(): Option[Seq[Seq[Double]]] = None

  /**
    * Get the improvement (positive) or damage (negative) due to each training row on a prediction
    *
    * @param actuals to assess the improvement or damage against
    * @return Sequence (over predictions) of sequence (over training rows) of influence
    */
  def getInfluenceScores(actuals: Seq[Any]): Option[Seq[Seq[Double]]] = None

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  def getGradient(): Option[Seq[Vector[Double]]] = None
}

trait RegressionResult extends PredictionResult[Double] {
  def getPredictionInterval(): Option[Seq[Double]] = None
  def getConfidenceInterval(): Option[Seq[Double]] = None
  def getBias(): Option[Seq[Double]] = None
  def getVariance(): Option[Seq[Double]] = None
  def getIrreducibleError(): Option[Seq[Double]] = None
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
