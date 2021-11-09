package io.citrine.lolo.stats

import io.citrine.lolo.linear.LinearRegressionLearner

import scala.util.Random

object StatsUtils {

  /** Compute the mean of a (weighted) vector, X */
  def mean(X: Seq[Double], weights: Option[Seq[Double]] = None): Double = {
    val actualWeights = weights.getOrElse(Seq.fill(X.length)(1.0))
    val totalWeight = actualWeights.sum
    require(totalWeight > 0.0, s"total weight must be positive, instead got $totalWeight")
    X.zip(actualWeights).map { case (x, w) => x * w }.sum / totalWeight
  }

  def median(X: Seq[Double]): Double = {
    val (lower, upper) = X.sorted.splitAt(X.size / 2)
    if (X.size % 2 == 0) (lower.last + upper.head) / 2.0 else upper.head
  }

  /** Compute the variance of a (weighted) vector, X, with dof degrees of freedom. */
  def variance(X: Seq[Double], weights: Option[Seq[Double]] = None, dof: Int = 0): Double = {
    val actualWeights = weights.getOrElse(Seq.fill(X.length)(1.0))
    val totalWeight = actualWeights.sum
    require(dof >= 0, s"degrees of freedom must be non-negative, instead got $dof")
    require(totalWeight > dof, s"Cannot compute variance on a sequence of weight $totalWeight with $dof degrees of freedom.")
    val mu = mean(X, Some(actualWeights))
    X.zip(actualWeights).map { case (x, w) => math.pow(x - mu, 2.0) * w }.sum / (totalWeight - dof)
  }

  /** Compute the (weighted) covariance between two vectors, X and Y, of the same length. */
  def covariance(X: Seq[Double], Y: Seq[Double], weights: Option[Seq[Double]] = None): Double = {
    require(X.length == Y.length, s"Cannot compute covariance between sequences of different lengths (lengths are ${X.length} and ${Y.length}).")
    val actualWeights = weights.getOrElse(Seq.fill(X.length)(1.0))
    val totalWeight = actualWeights.sum
    val muX = mean(X, Some(actualWeights))
    val muY = mean(Y, Some(actualWeights))
    (X, Y, actualWeights).zipped.map { case (x, y, w) => (x - muX) * (y - muY) * w }.sum / totalWeight
  }

  /** Compute the (weighted) correlation coefficient between two vectors, X and Y, of the same length. */
  def correlation(X: Seq[Double], Y: Seq[Double], weights: Option[Seq[Double]] = None): Double = {
    val actualWeights = weights.getOrElse(Seq.fill(X.length)(1.0))
    val sigma2X = variance(X, Some(actualWeights), dof = 0)
    val sigma2Y = variance(Y, Some(actualWeights), dof = 0)
    if (sigma2X == 0 || sigma2Y == 0) return 0.0
    covariance(X, Y, Some(actualWeights)) / math.sqrt(sigma2X * sigma2Y)
  }

  /** Given a univariate data set, construct a corresponding data set with the desired Pearson correlation coefficient.
    * The procedure is to generate new data randomly, calculate the residuals of a linear regression,
    * and then create a suitable linear combination of X and the residuals.
    * Source: https://stats.stackexchange.com/a/313138
    *
    * @param X    sequence of values
    * @param rho  desired Pearson correlation coefficient
    * @param rng  random number generator
    * @return     sequence of values that have desired correlation with X
    */
  def makeLinearCorrelatedData(X: Seq[Double], rho: Double, rng: Random = new Random()): Seq[Double] = {
    require(rho >= -1.0 && rho <= 1.0, "correlation coefficient must be between -1.0 and 1.0")
    val Y = Seq.fill(X.length)(rng.nextGaussian())
    val linearLearner = LinearRegressionLearner()
    val linearModel = linearLearner.train(X.zip(Y).map { case (x, y) => (Vector(x), y) } ).getModel()
    val yPred = linearModel.transform(X.map(Vector(_))).getExpected()
    val residuals = Y.zip(yPred).map { case (actual, predicted) => actual - predicted }
    val muX = mean(X)
    val stdX = math.sqrt(variance(X))
    val muResiduals = mean(residuals)
    val stdResiduals = math.sqrt(variance(residuals))
    X.zip(residuals).map { case (x, residual) => rho * stdResiduals * x + math.sqrt(1 - rho * rho) * stdX * residual}
  }

}
