package io.citrine.lolo.bags

import breeze.linalg.DenseMatrix
import io.citrine.lolo.PredictionResult
import org.slf4j.{Logger, LoggerFactory}

/**
  * Interface defining the return value of a [[BaggedModel]].
  *
  * This allows the implementation to depend on the number of simultaneous predictions,
  * which has performance implications.
  *
  * For background on the uncertainty calculation, see Wager, S.; Hastie, T and Efron, B. Confidence Intervals for
  * Random Forests: The Jackknife and Infinitesimal Jackknife. Journal of Machine Learning Research 15 (2014).
  */
trait BaggedResult[+T] extends PredictionResult[T] {

  /** The number of input rows that have been predicted on (NOT the number of bagged models). */
  def numPredictions: Int

  /** The predictions made by each of the bagged models. */
  def predictions: Seq[PredictionResult[T]]

  /**
    * Average the gradients from the models in the ensemble
    *
    * @return the gradient of each prediction as a vector of doubles
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = gradient

  private lazy val gradient = if (predictions.head.getGradient().isEmpty) {
    /* If the underlying model has no gradient, return None */
    None
  } else {
    val gradientsByPrediction: Seq[Seq[Vector[Double]]] = predictions.map(_.getGradient().get)
    val gradientsByInput: Seq[Seq[Vector[Double]]] = gradientsByPrediction.transpose
    Some(gradientsByInput.map { r =>
      r.toVector.transpose.map(_.sum / predictions.size)
    })
  }
}

object BaggedResult {

  val logger: Logger = LoggerFactory.getLogger(getClass)

  /**
    * Generate a matrix that is useful for computing (co)variance via jackknife after bootstrap (JaB).
    * The central term of the JaB calculation (Wager et. al. 2014, equation 6) is the difference between the
    * out-of-bag prediction on a point and the mean prediction on that point. If this is written as a single sum over bags,
    * then each point has a weight -1/B when it is in-bag and weight 1/|{N_{bi}=0}| - 1/B when it is out-of-bag (B is the number of bags).
    * This matrix encodes those weights, so when it is multiplied by the (# bags) x (# predictions) prediction matrix
    * we have a matrix of the \Delta terms from equation 6.
    *
    * @param Nib The (# training) x (# bags) matrix indicating how many times each training point is used in each bag
    */
  def getJackknifeAfterBootstrapMatrix(Nib: Vector[Vector[Int]]): DenseMatrix[Double] = {
    new DenseMatrix[Double](
      Nib.head.size,
      Nib.size,
      Nib.flatMap { v =>
        val itot = 1.0 / v.size // 1/B
        val icount = 1.0 / v.count(_ == 0) // 1/|{N_{bi}=0}|
        v.map(n => if (n == 0) icount - itot else -itot)
      }.toArray
    )
  }

  /**
    * Generate a matrix that is useful for computing (co)variance via infinitesimal jackknife (IJ).
    * The central term of the IJ calculation (Wager et. al. 2014, equation 5) is the covariance between the number of
    * times a training point appears in a bag and the prediction made by that bag.
    * This matrix encodes (N - \bar{N})/B (B is the number of bags), so that when it is multiplied by the
    *  (# bags) x (# predictions) prediction matrix, we have a matrix of the covariance terms from equation 5.
    *
    * @param Nib The (# training) x (# bags) matrix indicating how many times each training point is used in each bag
    */
  def getInfinitesimalJackknifeMatrix(Nib: Vector[Vector[Int]]): DenseMatrix[Double] = {
    new DenseMatrix[Double](
      Nib.head.size,
      Nib.size,
      Nib.flatMap { v =>
        val itot = 1.0 / v.size // 1/B
        val vtot = v.sum.toDouble / (v.size * v.size) // \bar{N} / B
        v.map(n => n * itot - vtot)
      }.toArray
    )
  }

  /**
    * Make sure the variance is non-negative
    *
    * The monte carlo bias correction is itself stochastic, so let's make sure the result is positive
    *
    * If the sum is positive, then great!  We're done.
    *
    * If the sum is <= 0.0, then the actual variance is likely quite small.  We know the variance should be at
    * least as large as the largest importance, since at least one training point will be important.
    * Therefore, let's just take the maximum importance, which should be a reasonable lower-bound of the variance.
    * Note that we could also sum the non-negative scores, but that could be biased upwards.
    *
    * If all of the scores are negative (which happens infrequently for very small ensembles), then we just need a scale.
    * The largest scale is the largest magnitude score, which is the absolute value of the minimum score.  When this
    * happens, then a larger ensemble should really be used!
    *
    * If all of the treePredictions are zero, then this will return zero.
    *
    * @param scores the monte-carlo corrected importance scores
    * @return A non-negative estimate of the variance
    */
  def rectifyEstimatedVariance(scores: Seq[Double]): Double = {
    val rawSum = scores.sum
    lazy val maxEntry = scores.max

    if (rawSum > 0) {
      rawSum
    } else if (maxEntry > 0) {
      // If the sum is negative,
      logger.warn(
        s"Sum of scores was negative; using the largest score as an estimate for the variance.  Please consider increasing the ensemble size."
      )
      maxEntry
    } else {
      logger.warn(
        s"All scores were negative; using the magnitude of the smallest score as an estimate for the variance.  It is highly recommended to increase the ensemble size."
      )
      -scores.min // equivalent to Math.abs(scores.min)
    }
  } ensuring (_ >= 0.0)

  /**
    * Make sure the scores are each non-negative
    *
    * The monte carlo bias correction is itself stochastic, so let's make sure the result is positive.
    * If the score was statistically consistent with zero, then we might subtract off the entire bias correction,
    * which results in the negative value.  Therefore, we can use the magnitude of the minimum as an estimate of the noise
    * level, and can simply set that as a floor.
    *
    * If all of the treePredictions are zero, then this will return a vector of zero
    *
    * @param scores the monte-carlo corrected importance scores
    * @return a vector of non-negative bias corrected scores
    */
  def rectifyImportanceScores(scores: Vector[Double]): Vector[Double] = {
    // this is a lower-bound on the noise level; note that it is strictly smaller than the correction
    val floor = Math.abs(scores.min)

    if (floor < 0.0) {
      logger.warn(s"Some importance scores were negative; rectifying.  Please consider increasing the ensemble size.")
    }
    scores.map(Math.max(floor, _))
  } ensuring (vec => vec.forall(_ >= 0.0))
}
