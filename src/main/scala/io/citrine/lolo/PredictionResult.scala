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
   * Get the standard deviation of the distribution of predicted observations, if possible
   *
   * Observations of the predicted variable are expected to have a stddev that matches this value.
   * This statistic is related to the https://en.wikipedia.org/wiki/Prediction_interval
   * It does not include estimated bias, if applicable
   */
  def getStdDevObs(): Option[Seq[Double]] = None

  /**
   * Get the expected error of the observations, if possible
   *
   * This statistic is related to the https://en.wikipedia.org/wiki/Prediction_interval
   * This statistic includes the contribution of the estimated bias.  E.g., for a normal distribution of predicted
   * means, the total error is sqrt(bias**2 + variance)
   */
  def getTotalErrorObs(): Option[Seq[Double]] = None

  /**
    * Get a quantile from the distribution of predicted observations, if possible
    *
    * Observations of the predicted variable are expected to have a distribution with this quantile.
    * This statistic is related to the https://en.wikipedia.org/wiki/Prediction_interval
   *  getObsQuantile(0.5) is a central statistic for the estimated bias, if the bias is estimated but not
   *  corrected.
    *
    * @param quantile to get, taken between 0.0 and 1.0 (i.e. not a percentile)
    */
  def getTotalErrorQuantileObs(quantile: Double): Option[Seq[Double]] = None

  /**
   * Get the expected error of the predicted mean observations, if possible
   *
   * The mean of a large sample of repeated observations are expected to have a stddev that matches this value.
   * This statistic is related to the https://en.wikipedia.org/wiki/Confidence_interval
   * This statistic includes the contribution of the estimated bias.  E.g., for a normal distribution of predicted
   * means, the total error is sqrt(bias**2 + variance)
   */
  def getTotalError(): Option[Seq[Double]] = None

  /**
   * Get a quantile from the distribution of predicted means, if possible
   *
   * The distribution for which these quantiles are computed could be biased, e.g. if the bias is estimated but not
   * corrected.
   * @param quantile to get, taken between 0.0 and 1.0 (i.e. not a percentile)
   */
  def getTotalErrorQuantile(quantile: Double): Option[Seq[Double]] = None

  /**
   * Get the standard deviation of the distribution of predicted mean observations, if possible
   *
   * The variation is the variation due to the finite size of the training data, which can be thought of as being
   * sampled from some training data distribution.
   * This statistic is related to the variance in the bias-variance trade-off
   * https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
   */
  def getStdDevMean(): Option[Seq[Double]] = None

  /**
   * Get a quantile from the distribution of predicted means, if possible
   *
   * The distribution for which these quantiles are computed should have zero-mean (i.e. no bias)
   * @param quantile to get, taken between 0.0 and 1.0 (i.e. not a percentile)
   */
  def getQuantileMean(quantile: Double): Option[Seq[Double]] = None

  /**
   * Get the estimated magnitude of the bias of each prediction, if possible
   *
   * The bias is signed and can be added to the prediction to improve accuracy.
   * See https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
   */
  def getBias(): Option[Seq[Double]] = None


  /**
   * Get the "uncertainty", which is the TotalError if non-observational and the StdDevObs if observational
   *
   * @param observational whether the uncertainty should account for observational uncertainty
   * @return uncertainty of each prediction
   */
  override def getUncertainty(observational: Boolean = true): Option[Seq[Any]] = {
    if (observational) {
      getTotalErrorObs()
    } else {
      getTotalError()
    }
  }

  def getQuantile(quantile: Double, observational: Boolean = true): Option[Seq[Double]] = {
    if (observational) {
      getTotalErrorQuantileObs(quantile)
    } else {
      getTotalErrorQuantile(quantile)
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
