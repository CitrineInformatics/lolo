package io.citrine.lolo

/**
  * Container for prediction results; must include expected values
  *
  * @tparam T the type of label variables being predicted
  */
trait PredictionResult[+T] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  def expected: Seq[T]

  /**
    * Get the "uncertainty" of the prediction
    *
    * For regression, this should be the TotalError if non-observational and the StdDevObs if observational
    *
    * @param observational whether the uncertainty should account for observational uncertainty
    * @return uncertainty of each prediction
    */
  def uncertainty(observational: Boolean = true): Option[Seq[Any]] = None

  /**
    * Get the training row scores for each prediction
    *
    * @return sequence (over predictions) of sequence (over training rows) of importances
    */
  def importanceScores: Option[Seq[Seq[Double]]] = None

  /**
    * Get the improvement (positive) or damage (negative) due to each training row on a prediction
    *
    * @param actuals to assess the improvement or damage against
    * @return Sequence (over predictions) of sequence (over training rows) of influence
    */
  def influenceScores(actuals: Seq[Any]): Option[Seq[Seq[Double]]] = None

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  def gradient: Option[Seq[Vector[Double]]] = None
}

/**
  * Additional regression-specific interface
  *
  * This interface is experimental and SHOULD BE REVIEWED before being merged into `master`.
  * In particular, an explanation of how the different methods relate to each other,
  * how predictive uncertainty is decomposed, and what the assumptions are
  * should be added, as these are currently not entirely clear.
  *
  * For example, does the interface assume that the predictions are the mean of a predictive distribution
  * (as opposed to, for example, the median, or the value with highest probability)?
  * Does it assume the predictive distribution to be normal?
  * Such assumptions are fine, but should be explicitly stated.
  */
trait RegressionResult extends PredictionResult[Double] {

  /**
    * Get the standard deviation of the distribution of predicted observations, if possible
    *
    * Observations of the predicted variable are expected to have a stddev that matches this value.
    * This statistic is related to the https://en.wikipedia.org/wiki/Prediction_interval
    * It does not include estimated bias, even if the regression result contains a bias estimate.
    */
  def stdDevObs: Option[Seq[Double]] = None

  /**
    * Get the expected error of the observations, if possible
    *
    * This statistic is related to the https://en.wikipedia.org/wiki/Prediction_interval
    * This statistic includes the contribution of the estimated bias.  E.g., for a normal distribution of predicted
    * means, the total error is sqrt(bias**2 + variance).
    */
  def totalErrorObs: Option[Seq[Double]] = None

  /**
    * Get a quantile from the distribution of predicted observations, if possible
    *
    * Observations of the predicted variable are inferred to have a distribution with this quantile.
    * This statistic is related to the https://en.wikipedia.org/wiki/Prediction_interval
    * getObsQuantile(0.5) is a central statistic for the estimated bias, if the bias is estimated but not
    * corrected.
    *
    * @param quantile to get, taken between 0.0 and 1.0 (i.e. not a percentile)
    */
  def totalErrorQuantileObs(quantile: Double): Option[Seq[Double]] = None

  /**
    * Get the expected error of the predicted mean observations, if possible
    *
    * The mean of a large sample of repeated observations are expected to have a root mean squared error of the mean
    * that matches this value.
    * This statistic is related to the https://en.wikipedia.org/wiki/Confidence_interval
    * This statistic includes the contribution of the estimated bias.  E.g., for a normal distribution of predicted
    * means, the total error is sqrt(bias**2 + variance)
    */
  def totalError: Option[Seq[Double]] = None

  /**
    * Get a quantile from the distribution of predicted means, if possible
    *
    * The distribution for which these quantiles are computed could be biased, e.g. if the bias is estimated but not
    * corrected.
    * @param quantile to get, taken between 0.0 and 1.0 (i.e. not a percentile)
    */
  def totalErrorQuantile(quantile: Double): Option[Seq[Double]] = None

  /**
    * Get the standard deviation of the distribution of predicted mean observations, if possible
    *
    * The variation is due to the finite size of the training data, which can be thought of as being
    * sampled from some training data distribution.
    * This statistic is related to the variance in the bias-variance trade-off
    * https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
    */
  def stdDevMean: Option[Seq[Double]] = None

  /**
    * Get a quantile from the distribution of predicted means, if possible
    *
    * The distribution for which these quantiles are computed should have zero-mean (i.e. no bias)
    * @param quantile to get, taken between 0.0 and 1.0 (i.e. not a percentile)
    */
  def quantileMean(quantile: Double): Option[Seq[Double]] = None

  /**
    * Get the estimated bias of each prediction, if possible.
    *
    * The bias is signed and can be subtracted from the prediction to improve accuracy.
    * See https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
    */
  def bias: Option[Seq[Double]] = None

  /**
    * Get the "uncertainty", which is the TotalError if non-observational and the StdDevObs if observational
    *
    * @param observational whether the uncertainty should account for observational uncertainty
    * @return uncertainty of each prediction
    */
  override def uncertainty(observational: Boolean = true): Option[Seq[Any]] = {
    if (observational) {
      totalErrorObs
    } else {
      totalError
    }
  }

  def quantile(quantile: Double, observational: Boolean = true): Option[Seq[Double]] = {
    if (observational) {
      totalErrorQuantileObs(quantile)
    } else {
      totalErrorQuantile(quantile)
    }
  }
}

/** Container for predictions made on multiple labels simultaneously. */
trait MultiTaskModelPredictionResult extends PredictionResult[Vector[Any]] {

  override def uncertainty(observational: Boolean = true): Option[Seq[Seq[Any]]] = None

  /**
    * Get the correlation coefficients between the predictions made on two labels.
    * Correlation coefficient is bounded between -1 and 1.
    * If either index is out of bounds or does not correspond to a real-valued label, then this method must return None.
    *
    * @param  i index of the first label
    * @param  j index of the second label
    * @param observational whether the uncertainty correlation should take observational noise into account
    * @return optional sequence of correlation coefficients between specified labels for each prediction
    */
  def uncertaintyCorrelation(i: Int, j: Int, observational: Boolean = true): Option[Seq[Double]] = None
}

/** A container that holds the predictions of several parallel models for multiple labels. */
case class ParallelModelsPredictionResult(expected: Seq[Vector[Any]]) extends MultiTaskModelPredictionResult
