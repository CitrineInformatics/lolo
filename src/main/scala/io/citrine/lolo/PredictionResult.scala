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
    * For regression, this should be a standard deviation for the output distribution.
    * If includeNoise, then it is a prediction interval; if not, then this is a confidence interval
    *
    * @param includeNoise whether the uncertainty should account for irreducible noise (i.e. a prediction interval)
    * @return uncertainty of each prediction
    */
  def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = None

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

/**
 * Additional regression-specific interface
 */
trait RegressionResult extends PredictionResult[Double] {
  /**
   * Get the 1-\sigma Prediction Interval for each prediction, if possible
   *
   * https://en.wikipedia.org/wiki/Prediction_interval
   * The prediction interval is typically the sum of the (square) bias, variance, and irreducible error
   */
  def getPredictionInterval(): Option[Seq[Double]] = None

  /**
   * Get the 1-\sigma Confidence Interval for each prediction, if possible
   * https://en.wikipedia.org/wiki/Confidence_interval
   */
  def getConfidenceInterval(): Option[Seq[Double]] = None

  /**
   * Get the estimated magnitude of the bias of each prediction, if possible
   *
   * See https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
   */
  def getBias(): Option[Seq[Double]] = None

  /**
   * Get the estimated magnitude of the sqrt of the variance of each prediction, if possible
   *
   * The sqrt is so that the bias, variance, and irreducible error have the same scale.
   * See https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
   */
  def getRootVariance(): Option[Seq[Double]] = None

  /**
   * Get the estimated magnitude of the irreducible noise of each prediction, if possible
   *
   * See https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
   */
  def getIrreducibleError(): Option[Seq[Double]] = None

  /**
   * Get the "uncertainty", which is the prediction interval if includeNoise and the confidence interval otherwise
   *
   * @param includeNoise whether the uncertainty should account for irreducible noise (i.e. a prediction interval)
   * @return uncertainty of each prediction
   */
  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
    if (includeNoise) {
      getPredictionInterval()
    } else {
      getConfidenceInterval()
    }
  }
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
