package io.citrine.lolo.validation

import java.util

import io.citrine.lolo.PredictionResult
import org.knowm.xchart.XYChart

import scala.collection.JavaConverters._
import scala.util.Random

/**
  * Real-valued metric on predictions of type T
  */
trait Merit[T] {

  /**
    * Apply the metric to a prediction result and set of ground-truth values
    *
    * @return the value of the metric
    */
  def evaluate(predictionResult: PredictionResult[T], actual: Seq[T]): Double

  /**
    * Estimate the metric and the uncertainty in the metric over batches of predicted and ground-truth values
    *
    * @param pva predicted-vs-actual data as an iterable over [[PredictionResult]] and ground-truth tuples
    * @return the estimate of the metric value and the uncertainty in that estimate
    */
  def estimate(pva: Iterable[(PredictionResult[T], Seq[T])]): (Double, Double) = {
    val samples = pva.map { case (prediction, actual) => evaluate(prediction, actual) }
    val mean: Double = samples.sum / samples.size
    val variance: Double = (samples.size / (samples.size - 1)) * samples.map(x => Math.pow(x - mean, 2)).sum / samples.size
    (mean, Math.sqrt(variance / samples.size))
  }
}

/**
  * SSIA
  */
case object RootMeanSquareError extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
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
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    val averageActual = actual.sum / actual.size
    val sumOfSquares = actual.map(x => Math.pow(x - averageActual, 2)).sum
    val sumOfResiduals = predictionResult.getExpected().zip(actual).map { case (x, y) => Math.pow(x - y, 2.0) }.sum
    1.0 - sumOfResiduals / sumOfSquares
  }
}

/**
  * The fraction of predictions that fall within the predicted uncertainty
  */
case object StandardConfidence extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    if (predictionResult.getUncertainty().isEmpty) return 0.0

    (predictionResult.getExpected(), predictionResult.getUncertainty().get, actual).zipped.count {
      case (x, sigma: Double, y) => Math.abs(x - y) < sigma
    } / predictionResult.getExpected().size.toDouble
  }
}

/**
  * Root mean square of (the error divided by the predicted uncertainty)
  */
case class StandardError(rescale: Double = 1.0) extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
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
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    val predictedUncertaintyActual: Seq[(Double, Double, Double)] = (
      predictionResult.getExpected(),
      predictionResult.getUncertainty().get.asInstanceOf[Seq[Double]],
      actual
    ).zipped.toSeq

    val ideal = predictedUncertaintyActual.map { case (_, uncertainty, actual) =>
      val error = Random.nextGaussian() * uncertainty
      (actual + error, uncertainty, actual)
    }

    /*
    val sigmaMean = pua.map(_._2).sum / pua.size
    val sigmaVar = pua.map(x => Math.pow(x._2 - sigmaMean, 2)).sum / pua.size
    val model = Math.sqrt(2 / (Math.PI - 2)) * Math.sqrt(sigmaVar) / sigmaMean
    val estimate = computeFromPredictedUncertaintyActual(baseline)
    // println(model, estimate)
    */

    computeFromPredictedUncertaintyActual(predictedUncertaintyActual) / computeFromPredictedUncertaintyActual(ideal)
  }

  /**
    * Covariance(X, Y) / Sqrt(Var(X) * Var(Y)), where X is predicted uncertainty and Y is magnitude of error
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

object Merit {

  /**
    * Estimate a set of named metrics by applying them to multiple sets of predictions and actual values
    *
    * The uncertainty in the estimate of each metric is calculated by looking at the variance across the batches
    *
    * @param pva     predicted-vs-actual data in a series of batches
    * @param metrics to apply to the predicted-vs-actual data
    * @return map from the metric name to its (value, uncertainty)
    */
  def estimateMerits[T](
                          pva: Iterable[(PredictionResult[T], Seq[T])],
                          metrics: Map[String, Merit[T]]
                        ): Map[String, (Double, Double)] = {

    pva.flatMap { case (predictions, actual) =>
      // apply all the metrics to the batch at the same time so the batch can fall out of memory
      metrics.mapValues(f => f.evaluate(predictions, actual)).toSeq
    }.groupBy(_._1).mapValues { x =>
      val metricResults = x.map(_._2)
      val mean = metricResults.sum / metricResults.size
      val variance = metricResults.map(y => Math.pow(y - mean, 2)).sum / metricResults.size
      (mean, Math.sqrt(variance / metricResults.size))
    }
  }

  /**
    * Compute metrics as a function of a parameter, given a builder that takes the parameter to predicted-vs-actual data
    *
    * @param parameterName   name of the parameter that's being scanned over
    * @param parameterValues values of the parameter to try
    * @param metrics         to apply at each parameter value
    * @param logScale        whether the parameters should be plotted on a log scale
    * @param pvaBuilder      function that takes the parameter to predicted-vs-actual data
    * @return an [[XYChart]] that plots the metrics vs the parameter value
    */
  def plotMeritScan[T](
                         parameterName: String,
                         parameterValues: Seq[Double],
                         metrics: Map[String, Merit[T]],
                         logScale: Boolean = false,
                         yMin: Option[Double] = None,
                         yMax: Option[Double] = None
                       )(
                         pvaBuilder: Double => Iterable[(PredictionResult[T], Seq[T])]
                       ): XYChart = {

    val seriesData: Map[String, util.ArrayList[Double]] = metrics.flatMap { case (name, _) =>
      Seq(
        name -> new util.ArrayList[Double](),
        s"${name}_err" -> new util.ArrayList[Double]()
      )
    }

    parameterValues.foreach { param =>
      val pva = pvaBuilder(param)
      val metricResults = Merit.estimateMerits(pva, metrics)
      metricResults.foreach { case (name, (mean, err)) =>
        seriesData(name).add(mean)
        seriesData(s"${name}_err").add(err)
      }
    }
    val chart = new XYChart(500, 500)
    chart.setTitle(s"Scan over $parameterName")
    chart.setXAxisTitle(parameterName)
    metrics.map { case (name, _) =>
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
