package io.citrine.lolo.stats.correlations

import breeze.linalg.{*, sum, DenseMatrix, DenseVector}

/**
  * Object for computing distance correlations
  *
  * See https://en.wikipedia.org/wiki/Distance_correlation
  * Created by maxhutch on 12/23/16.
  */
object DistanceCorrelation {

  /**
    * Double centered differnce matrix
    *
    * @param x        sequence to take differences of
    * @param distance function
    * @tparam T of the sequence
    * @return a double centered distance matrix
    */
  def doubleCenter[T](x: Seq[T], distance: (T, T) => Double): DenseMatrix[Double] = {
    val pairDistances: DenseMatrix[Double] = DenseMatrix.tabulate(x.size, x.size) {
      case (i, j) =>
        distance(x(i), x(j))
    }
    val grandMean = sum(pairDistances) / (x.size * x.size)
    val colSum: DenseVector[Double] = sum(pairDistances(*, ::))
    val colMean: DenseVector[Double] = colSum *:* (1.0 / x.size)
    DenseMatrix.tabulate(x.size, x.size) {
      case (i, j) =>
        pairDistances(i, j) - colMean(i) - colMean(j) + grandMean
    }
  }

  /**
    * Sample distance covariance
    *
    * @param x        first sequence
    * @param y        second sequence
    * @param distance distance function
    * @tparam T of the sequences
    * @return the sample distance covariance
    */
  def distanceCovariance[T](x: Seq[T], y: Seq[T], distance: (T, T) => Double): Double = {
    val A: DenseMatrix[Double] = doubleCenter(x, distance)
    val B: DenseMatrix[Double] = doubleCenter(y, distance)
    Math.sqrt(sum(A *:* B) / (x.size * x.size))
  }

  /**
    * Distance correlation function, based on distance covariance
    *
    * @param x        first sequence
    * @param y        second sequence
    * @param distance function
    * @tparam T of the sequences
    * @return the distance correlation (dCorr)
    */
  def distanceCorrelation[T](x: Seq[T], y: Seq[T], distance: (T, T) => Double): Double = {
    val cov = distanceCovariance(x, y, distance)
    val varx = distanceCovariance(x, x, distance)
    val vary = distanceCovariance(y, y, distance)
    cov / Math.sqrt(varx * vary)
  }
}
