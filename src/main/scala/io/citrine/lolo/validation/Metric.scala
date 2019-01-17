package io.citrine.lolo.validation

import java.util

import scala.collection.JavaConverters._
import io.citrine.lolo.PredictionResult
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation
import org.knowm.xchart.BitmapEncoder.BitmapFormat
import org.knowm.xchart.{BitmapEncoder, CategoryChart, CategoryChartBuilder, QuickChart, SwingWrapper, XYChart}

import scala.util.Random

trait Metric[T] {

  def evaluate(predictionResult: PredictionResult[T], actual: Seq[T]): Double

  def estimate(pva: Iterable[(PredictionResult[T], Seq[T])]): (Double, Double) = {
    val samples = pva.map{case (prediction, actual) => evaluate(prediction, actual)}
    val mean: Double = samples.sum / samples.size
    val variance: Double = samples.map(x => Math.pow(x - mean, 2)).sum / samples.size
    (mean, Math.sqrt(variance / samples.size))
  }
}

case object RootMeanSquareError extends Metric[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    Math.sqrt(
      (predictionResult.getExpected(), actual).zipped.map{
        case (x, y) => Math.pow(x - y, 2)
      }.sum / predictionResult.getExpected().size
    )
  }
}

case object CoefficientOfDetermination extends Metric[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    val averageActual = actual.sum / actual.size
    val sumOfSquares = actual.map(x => Math.pow(x - averageActual, 2)).sum
    val sumOfResiduals = predictionResult.getExpected().zip(actual).map{case (x, y) => Math.pow(x - y, 2.0)}.sum
    1.0 - sumOfResiduals / sumOfSquares
  }
}

case object StandardConfidence extends Metric[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    if (predictionResult.getUncertainty().isEmpty) return 0.0

    (predictionResult.getExpected(), predictionResult.getUncertainty().get, actual).zipped.count{
      case (x, sigma: Double, y) => Math.abs(x - y) < sigma
    } / predictionResult.getExpected().size.toDouble
  }
}

case object StandardError extends Metric[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    if (predictionResult.getUncertainty().isEmpty) return Double.PositiveInfinity
    val standardized = (predictionResult.getExpected(), predictionResult.getUncertainty().get, actual).zipped.map{
       case (x, sigma: Double, y) => (x - y) / sigma
    }
    Math.sqrt(standardized.map(Math.pow(_,2.0)).sum / standardized.size)
  }
}

case object UncertaintyCorrelation extends Metric[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double]): Double = {
    val pua = (predictionResult.getExpected(), predictionResult.getUncertainty().get.asInstanceOf[Seq[Double]], actual).zipped.toSeq
    val baseline = pua.map{case (_, u, a) =>
        val error = Random.nextGaussian() * u
      (a + error, u, a)
    }
    computeFromPredictedUncertaintyActual(pua) / computeFromPredictedUncertaintyActual(baseline)
  }

  def computeFromPredictedUncertaintyActual(pua: Seq[(Double, Double, Double)]): Double = {
    val error = pua.map{case (p, u, a) => Math.abs(p - a)}
    val sigma = pua.map(_._2)

    val meanError = error.sum / error.size
    val varError = error.map(x => Math.pow(x - meanError, 2.0)).sum / error.size
    val meanSigma = sigma.sum / sigma.size
    val varSigma = sigma.map(x => Math.pow(x - meanSigma, 2.0)).sum / sigma.size

    val covar = error.zip(sigma).map{case (x, y) => (x - meanError) * (y - meanSigma)}.sum / sigma.size
    covar / Math.sqrt(varError * varSigma)
  }
}

object Metric {

  def estimateMetrics[T](
                          pva: Iterable[(PredictionResult[T], Seq[T])],
                          metrics: Map[String, Metric[T]]
                        ): Map[String, (Double, Double)] = {
    pva.flatMap{ case (predictions, actual) =>
      // apply all the metrics to the batch at the same time so the batch can fall out of memory
      metrics.mapValues(f => f.evaluate(predictions, actual)).toSeq
    }.groupBy(_._1).mapValues{x =>
      val metricResults = x.map(_._2)
      val mean = metricResults.sum / metricResults.size
      val variance = metricResults.map(y => Math.pow(y - mean, 2)).sum / metricResults.size
      (mean, Math.sqrt(variance / metricResults.size))
    }
  }

  def scanMetrics[T](
                      parameterName: String,
                      parameterValues: Seq[Double],
                      metrics: Map[String, Metric[T]],
                      logScale: Boolean = false
                    )(
                      pvaBuilder: Double => Iterable[(PredictionResult[T], Seq[T])],
                    ): XYChart = {

    val seriesData: Map[String, util.ArrayList[Double]] = metrics.flatMap{case (name, _) =>
        Seq(
          name -> new util.ArrayList[Double](),
          s"${name}_err" -> new util.ArrayList[Double]()
        )
    }

    parameterValues.foreach{param =>
      val pva = pvaBuilder(param)
      val metricResults = Metric.estimateMetrics(pva, metrics)
      metricResults.foreach{ case (name, (mean, err)) =>
        seriesData(name).add(mean)
        seriesData(s"${name}_err").add(err)
      }
    }
    val chart = new XYChart(500, 500);
    chart.setTitle(s"Scan over $parameterName")
    chart.setXAxisTitle(parameterName)
    metrics.map{case (name, _) =>
        chart.addSeries(name, parameterValues.toArray, seriesData(name).asScala.toArray, seriesData(s"${name}_err").asScala.toArray)
    }
    if (logScale) {
      chart.getStyler.setXAxisLogarithmic(true)
    }

    chart
  }
}