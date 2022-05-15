package io.citrine.lolo.stats

object StatsUtils {

  /**
    * Compute the range of the data.
    *
    * @param X to compute the range of
    * @return the range, which is zero if there are no values
    */
  def range(X: Seq[Double]): Double = {
    if (X.nonEmpty) {
      X.max - X.min
    } else {
      0.0
    }
  }

  /** Compute the mean of a (weighted) vector, X */
  def mean(X: Seq[Double], weights: Option[Seq[Double]] = None): Double = {
    val actualWeights = weights.getOrElse(Seq.fill(X.length)(1.0))
    val totalWeight = actualWeights.sum
    require(totalWeight > 0.0, s"total weight must be positive, instead got $totalWeight")
    X.zip(actualWeights).map { case (x, w) => x * w }.sum / totalWeight
  }

  /** Compute the variance of a (weighted) vector, X, with dof degrees of freedom. */
  def variance(X: Seq[Double], weights: Option[Seq[Double]] = None, dof: Int = 0): Double = {
    val actualWeights = weights.getOrElse(Seq.fill(X.length)(1.0))
    val totalWeight = actualWeights.sum
    require(dof >= 0, s"degrees of freedom must be non-negative, instead got $dof")
    require(
      totalWeight > dof,
      s"Cannot compute variance on a sequence of weight $totalWeight with $dof degrees of freedom."
    )
    val mu = mean(X, Some(actualWeights))
    X.zip(actualWeights).map { case (x, w) => math.pow(x - mu, 2.0) * w }.sum / (totalWeight - dof)
  }

  /** Compute the (weighted) covariance between two vectors, X and Y, of the same length. */
  def covariance(X: Seq[Double], Y: Seq[Double], weights: Option[Seq[Double]] = None): Double = {
    require(
      X.length == Y.length,
      s"Cannot compute covariance between sequences of different lengths (lengths are ${X.length} and ${Y.length})."
    )
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
}
