package io.citrine.lolo.validation

import java.util
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.bags.CorrelationMethods.CorrelationMethod
import io.citrine.lolo.bags.MultiTaskBaggedResult
import io.citrine.lolo.stats.StatsUtils
import org.knowm.xchart.XYChart
import breeze.numerics.erfinv

import scala.collection.JavaConverters._
import scala.util.Random

/**
  * Real-valued figure of merit on predictions of type T
  */
trait Merit[T] {

  /**
    * Apply the figure of merti to a prediction result and set of ground-truth values
    *
    * @return the value of the figure of merit
    */
  def evaluate(predictionResult: PredictionResult[T], actual: Seq[T], rng: Random = Random): Double

  /**
    * Estimate the merit and the uncertainty in the merit over batches of predicted and ground-truth values
    *
    * @param pva predicted-vs-actual data as an iterable over [[PredictionResult]] and ground-truth tuples
    * @return the estimate of the merit value and the uncertainty in that estimate
    */
  def estimate(pva: Iterable[(PredictionResult[T], Seq[T])], rng: Random = Random): (Double, Double) = {
    val samples = pva.map { case (prediction, actual) => evaluate(prediction, actual, rng) }
    val mean: Double = samples.sum / samples.size
    val variance: Double = (samples.size / (samples.size - 1)) * samples.map(x => Math.pow(x - mean, 2)).sum / samples.size
    (mean, Math.sqrt(variance / samples.size))
  }
}

/**
  * Square root of the mean square error. For an unbiased estimator, this is equal to the standard deviation of the difference between predicted and actual values.
  */
case object RootMeanSquareError extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    Math.sqrt(
      (predictionResult.getExpected(), actual).zipped.map {
        case (x, y) => Math.pow(x - y, 2)
      }.sum / predictionResult.getExpected().size
    )
  }
}

/**
  * R2 = 1 - MSE(y) / Var(y), where y is the predicted variable
  */
case object CoefficientOfDetermination extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    val averageActual = actual.sum / actual.size
    val sumOfSquares = actual.map(x => Math.pow(x - averageActual, 2)).sum
    val sumOfResiduals = predictionResult.getExpected().zip(actual).map { case (x, y) => Math.pow(x - y, 2.0) }.sum
    1.0 - sumOfResiduals / sumOfSquares
  }
}

/**
  * The fraction of predictions that fall within the predicted uncertainty
  */
case class StandardConfidence(observational: Boolean = true, coverageLevel: Double = 0.683) extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    if (predictionResult.getUncertainty().isEmpty) return 0.0
    val numSigmas = erfinv(coverageLevel) * math.sqrt(2)

    (predictionResult.getExpected(), predictionResult.getUncertainty(observational = observational).get, actual).zipped.count {
      case (x, sigma: Double, y) => Math.abs(x - y) < sigma * numSigmas
    } / predictionResult.getExpected().size.toDouble
  }
}

/**
  * Root mean square of (the error divided by the predicted uncertainty)
  */
case class StandardError(rescale: Double = 1.0) extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    if (predictionResult.getUncertainty().isEmpty) return Double.PositiveInfinity
    val standardized = (predictionResult.getExpected(), predictionResult.getUncertainty().get, actual).zipped.map {
      case (x, sigma: Double, y) => (x - y) / sigma
    }
    rescale * Math.sqrt(standardized.map(Math.pow(_, 2.0)).sum / standardized.size)
  }
}

/**
  * Measure of the correlation between the predicted uncertainty and error magnitude
  *
  * This is expressed as a ratio of correlation coefficients.  The numerator is the correlation coefficient of the
  * predicted uncertainty and the actual error magnitude.  The denominator is the correlation coefficient of the
  * predicted uncertainty and the ideal error distribution.  That is:
  * let X be the predicted uncertainty and Y := N(0, x) be the ideal error distribution about each
  * predicted uncertainty x.  It is the correlation coefficient between X and Y
  * In the absence of a closed form for that coefficient, it is model empirically by drawing from N(0, x) to produce
  * an "ideal" error series from which the correlation coefficient can be estimated.
  */
case object UncertaintyCorrelation extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    val predictedUncertaintyActual: Seq[(Double, Double, Double)] = (
      predictionResult.getExpected(),
      predictionResult.getUncertainty().get.asInstanceOf[Seq[Double]],
      actual
    ).zipped.toSeq

    val ideal = predictedUncertaintyActual.map { case (_, uncertainty, actual) =>
      val error = rng.nextGaussian() * uncertainty
      (actual + error, uncertainty, actual)
    }

    computeFromPredictedUncertaintyActual(predictedUncertaintyActual) / computeFromPredictedUncertaintyActual(ideal)
  }

  /**
    * Covariance(X, Y) / Sqrt(Var(X) * Var(Y)), where X is predicted uncertainty and Y is magnitude of error
    * @param pua  predicted, uncertainty, and actual
    */
  def computeFromPredictedUncertaintyActual(
                                             pua: Seq[(Double, Double, Double)]
                                           ): Double = {
    val error = pua.map { case (p, _, a) => Math.abs(p - a) }
    val sigma = pua.map(_._2)

    val meanError = error.sum / error.size
    val varError = error.map(x => Math.pow(x - meanError, 2.0)).sum / error.size
    val meanSigma = sigma.sum / sigma.size
    val varSigma = sigma.map(x => Math.pow(x - meanSigma, 2.0)).sum / sigma.size

    val covar = error.zip(sigma).map { case (x, y) => (x - meanError) * (y - meanSigma) }.sum / sigma.size
    covar / Math.sqrt(varError * varSigma)
  }
}

/**
  * Negative Log Probability Density (NLPD) in two dimensions
  * NLPD is calculated for each point and the median value is returned.
  *
  * @param i index of the first label to be compared
  * @param j index of the second label to be compared
  * @param method method to calculate correlation coefficient
  * @param observational whether or not to calculate the observational uncertainty
  */
case class NegativeLogProbabilityDensity2d(i: Int, j: Int, method: CorrelationMethod, observational: Boolean) extends Merit[Seq[Any]] {
  override def evaluate(predictionResult: PredictionResult[Seq[Any]], actual: Seq[Seq[Any]], rng: Random): Double = {
    val pvas = PredictedVsActualTwoDimensions.makePva2d(predictionResult, actual, i, j, method, observational)
    val nlpd = NLPD(pvas)
    StatsUtils.median(nlpd)
  }

  /**
    * Bivariate negative log probability density calculation
    * log(2 * pi * sigma_x * sigma_y * sqrt(1 - rho**2)) + r**2/2, where r is the Mahalanobis distance
    * @param pvas sequence of bivariate predicted-vs-actual points
    * @return sequence of NLPD value for each prediction
    */
  def NLPD(pvas: Seq[PredictedVsActualTwoDimensions]): Seq[Double] = {
    pvas.map { pva =>
      val term1 = math.log(2 * math.Pi * pva.sigmaX * pva.sigmaY * math.sqrt(1 - math.pow(pva.rho, 2.0)))
      val term2 = 0.5 * pva.mahalanobisSquared
      term1 + term2
    }
  }
}

/**
  * Standard confidence in two dimensions. This is the fraction of predictions that fall within a given uncertainty band.
  * Uses the Mahalanobis distance to calculate the distance from the prediction to the actual point.
  *
  * @param i index of the first label to be compared
  * @param j index of the second label to be compared
  * @param method method to calculate correlation coefficient
  * @param observational whether or not to calculate the observational uncertainty
  * @param confidenceLevel the fraction of predictions that are expected to be within this distance.
  *                        By default it is 68.3%, corresponding to one standard deviation in one dimension.
  */
case class StandardConfidence2d(
                                 i: Int,
                                 j: Int,
                                 method: CorrelationMethod,
                                 observational: Boolean,
                                 confidenceLevel: Double = 0.683
                               ) extends Merit[Seq[Any]] {
  override def evaluate(predictionResult: PredictionResult[Seq[Any]], actual: Seq[Seq[Any]], rng: Random): Double = {
    val pvas = PredictedVsActualTwoDimensions.makePva2d(predictionResult, actual, i, j, method, observational)
    // The CDF of a bivariate normal distribution is 1 - exp(-r^2/2), where r is the Mahalanobis distance.
    // Therefore for a given desired probability, p, the corresponding distance is r^2 = -2 * ln(1 - p)
    val cutoffDistanceSquared = -2 * math.log(1 - confidenceLevel)
    pvas.count(_.mahalanobisSquared < cutoffDistanceSquared).toDouble / pvas.length
  }
}

object Merit {

  /**
    * Estimate a set of named merits by applying them to multiple sets of predictions and actual values
    *
    * The uncertainty in the estimate of each merit is calculated by looking at the variance across the batches
    *
    * @param pva     predicted-vs-actual data in a series of batches
    * @param merits  to apply to the predicted-vs-actual data
    * @return map from the merit name to its (value, uncertainty)
    */
  def estimateMerits[T](
                          pva: Iterator[(PredictionResult[T], Seq[T])],
                          merits: Map[String, Merit[T]],
                          rng: Random = Random
                        ): Map[String, (Double, Double)] = {

    pva.flatMap { case (predictions, actual) =>
      // apply all the merits to the batch at the same time so the batch can fall out of memory
      merits.mapValues(f => f.evaluate(predictions, actual, rng)).toSeq
    }.toIterable.groupBy(_._1).mapValues { x =>
      val meritResults = x.map(_._2)
      val mean = meritResults.sum / meritResults.size
      val variance = meritResults.map(y => Math.pow(y - mean, 2)).sum / meritResults.size
      (mean, Math.sqrt(variance / meritResults.size))
    }.toMap
  }

  /**
    * Compute merits as a function of a parameter, given a builder that takes the parameter to predicted-vs-actual data
    *
    * @param parameterName   name of the parameter that's being scanned over
    * @param parameterValues values of the parameter to try
    * @param merits          to apply at each parameter value
    * @param logScale        whether the parameters should be plotted on a log scale
    * @param pvaBuilder      function that takes the parameter to predicted-vs-actual data
    * @param rng             random number generator to use
    * @return an [[XYChart]] that plots the merits vs the parameter value
    */
  def plotMeritScan[T](
                         parameterName: String,
                         parameterValues: Seq[Double],
                         merits: Map[String, Merit[T]],
                         logScale: Boolean = false,
                         yMin: Option[Double] = None,
                         yMax: Option[Double] = None,
                         rng: Random = Random
                       )(
                         pvaBuilder: Double => Iterator[(PredictionResult[T], Seq[T])]
                       ): XYChart = {

    val seriesData: Map[String, util.ArrayList[Double]] = merits.flatMap { case (name, _) =>
      Seq(
        name -> new util.ArrayList[Double](),
        s"${name}_err" -> new util.ArrayList[Double]()
      )
    }

    parameterValues.foreach { param =>
      val pva = pvaBuilder(param)
      val meritResults = Merit.estimateMerits(pva, merits, rng)
      meritResults.foreach { case (name, (mean, err)) =>
        seriesData(name).add(mean)
        seriesData(s"${name}_err").add(err)
      }
    }
    val chart = new XYChart(900, 600)
    chart.setTitle(s"Scan over $parameterName")
    chart.setXAxisTitle(parameterName)
    merits.map { case (name, _) =>
      chart.addSeries(name, parameterValues.toArray, seriesData(name).asScala.toArray, seriesData(s"${name}_err").asScala.toArray)
    }

    if (logScale) {
      chart.getStyler.setXAxisLogarithmic(true)
    }

    yMin.foreach(min => chart.getStyler.setYAxisMin(min))
    yMax.foreach(max => chart.getStyler.setYAxisMax(max))

    chart
  }
}

/**
  * A single predicted-vs-actual point in two real-valued dimensions.
  * The covariance matrix of the prediction has diagonals sigmaX**2 and sigmaY**2, and off-diagonals rho*sigmaX*sigmaY
  *
  * @param dx the error along the first dimension
  * @param dy the error along the second dimension
  * @param sigmaX the uncertainty along the first dimension
  * @param sigmaY the uncertainty along the second dimension
  * @param rho the correlation coefficient
  */
case class PredictedVsActualTwoDimensions(
                                         dx: Double,
                                         dy: Double,
                                         sigmaX: Double,
                                         sigmaY: Double,
                                         rho: Double
                                         ) {
  private lazy val normX = dx / sigmaX
  private lazy val normY = dy / sigmaY
  private lazy val rhoSquared = math.pow(rho, 2.0)

  /**
    * Mahalanobis distance is a generalization of standardized distance for a multivariate normal distribution.
    * It corresponds to the Euclidean distance if space is rescaled such that the principal axes of the ellipsoid
    * formed by the covariance matrix each have length 1. In one dimension, this is (x - mu) / sigma.
    * In two dimensions, it is ((dx / sigmaX)**2 + (dy / sigmaY)**2 - 2 * rho * dx * dy / (sigmaX * sigmaY)) / (1 - rho**2)
    */
  def mahalanobisSquared: Double = {
    (math.pow(normX, 2.0) + math.pow(normY, 2.0) - 2 * rho * normX * normY) / (1 - rhoSquared)
  }
}

object PredictedVsActualTwoDimensions {

  /**
    * Convert a prediction result into a sequence of PredictedVsActualTwoDimensions objects, which are easier to compute metrics on.
    *
    * @param predictionResult multivariate prediction result
    * @param actual multivariate ground-truth values
    * @param i index of the first label to be compared
    * @param j index of the second label to be compared
    * @param method method to calculate correlation coefficient
    * @param observational whether or not to calculate the observational uncertainty
    * @return a case class containing the error and uncertainty for each predicted-actual pair
    */
  def makePva2d(
                 predictionResult: PredictionResult[Seq[Any]],
                 actual: Seq[Seq[Any]],
                 i: Int,
                 j: Int,
                 method: CorrelationMethod,
                 observational: Boolean
               ): Seq[PredictedVsActualTwoDimensions] = {
    val allPredictions = predictionResult.getExpected()
    // get predictions
    val predictionsI = extractComponentByIndex(allPredictions, i)
    val predictionsJ = extractComponentByIndex(allPredictions, j)
    // get actual
    val actualI = extractComponentByIndex(actual, i)
    val actualJ = extractComponentByIndex(actual, j)
    // subtract predicted - actual to get error
    val errorI = predictionsI.zip(actualI).map(pa => pa._1 - pa._2)
    val errorJ = predictionsJ.zip(actualJ).map(pa => pa._1 - pa._2)
    // get terms of covariance matrix
    val allSigma = predictionResult.asInstanceOf[MultiTaskBaggedResult].getUncertainty(observational).get
    val sigmaI = extractComponentByIndex(allSigma, i, Some(0.0))
    val sigmaJ = extractComponentByIndex(allSigma, j, Some(0.0))
    val correlation = predictionResult.asInstanceOf[MultiTaskBaggedResult].getUncertaintyCorrelationBuffet(i, j, method).get
    Seq(errorI, errorJ, sigmaI, sigmaJ, correlation).transpose.map { values =>
      PredictedVsActualTwoDimensions(values.head, values(1), values(2), values(3), values(4))
    }
  }

  /**
    * A convenience method for slicing a two dimensional data structure
    *
    * @param data a sequence of sequences
    * @param index to slice (must correspond to real-valued data)
    * @param default default to use in case the values are Option[Double]
    * @return
    */
  def extractComponentByIndex(data: Seq[Seq[Any]], index: Int, default: Option[Double] = None): Seq[Double] = {
    data.map { row =>
      row(index) match {
        case x: Double => x
        case x: Option[Double] => x.getOrElse(default.get)
      }
    }
  }

}
