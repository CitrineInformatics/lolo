package io.citrine.lolo.validation

import io.citrine.lolo.PredictionResult
import org.apache.commons.math3.distribution.CauchyDistribution
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle
import org.knowm.xchart._
import org.knowm.xchart.internal.chartpart.Chart

/**
  * Visualization on predicted vs actual data of type T
  */
trait Visualization[T] {

  /**
    * Produce a visualization, as a [[Chart]], from predicted-vs-actual data
    */
  def visualize(data: Iterable[(PredictionResult[T], Seq[T])]): Chart[_, _]
}

/**
  * Histogram of the error divided by the predicted uncertainty
  *
  * Gaussian and Cauchy fits are preformed via quantiles:
  *  - standard deviation is taken as the 68th percentile standard error
  *  - gamma is taken as the 50th percentile standard error
  *
  * @param nBins       number of bins in the histogram
  * @param range       of the horizontal axis, e.g. x \in [-range/2, range/2]
  * @param fitGaussian whether to fit and plot a Gaussian distribution
  * @param fitCauchy   whether to fit and plot a Cauchy distribution
  */
case class StandardResidualHistogram(
    nBins: Int = 128,
    range: Double = 8.0,
    fitGaussian: Boolean = true,
    fitCauchy: Boolean = true
) extends Visualization[Double] {

  override def visualize(data: Iterable[(PredictionResult[Double], Seq[Double])]): CategoryChart = {
    val pua: Seq[(Double, Double, Double)] = data.flatMap {
      case (predictions, actual) =>
        predictions.expected
          .lazyZip(predictions.uncertainty().get.asInstanceOf[Seq[Double]])
          .lazyZip(actual)
          .toSeq
    }.toSeq

    val standardErrors = pua.map { case (predicted, sigma, actual) => (predicted - actual) / sigma }.sorted

    val bins = (-nBins / 2 until nBins / 2).map { idx =>
      (range * idx.toDouble / nBins, range * (idx.toDouble + 1) / nBins)
    }

    val counts = bins.map {
      case (min, max) =>
        ((min + max) / 2.0, standardErrors.count(x => x >= min && x < max) / (standardErrors.size * (max - min)))
    }

    val chart: CategoryChart = new CategoryChartBuilder().build()
    chart.addSeries("data", counts.map(_._1).toArray, counts.map(_._2.toDouble).toArray)

    if (fitGaussian) {
      val normalVar = standardErrors.drop((standardErrors.size * 0.68).toInt).head
      val normalSeries =
        counts.map(_._1).map(x => Math.exp(-x * x / (2 * normalVar)) / Math.sqrt(2 * Math.PI * normalVar))
      chart.addSeries(f"Normal(0, ${Math.sqrt(normalVar)}%6.3f)", counts.map(_._1).toArray, normalSeries.toArray)
    }
    if (fitCauchy) {
      val gamma = standardErrors.drop((standardErrors.size * 0.5).toInt).head
      val cauchy1 = new CauchyDistribution(0.0, gamma)
      val cauchySeries = counts.map(_._1).map(x => cauchy1.density(x))
      chart.addSeries(f"Cauchy(0, ${gamma}%6.3f)", counts.map(_._1).toArray, cauchySeries.toArray)
    }

    chart.setTitle("(predicted - actual) / (predicted uncertainty)")
    chart.setYAxisTitle("probability density")

    chart
  }
}

/**
  * Plot the predicted value vs the actual value, with predicted uncertainty as error bars
  */
case class PredictedVsActual() extends Visualization[Double] {

  override def visualize(data: Iterable[(PredictionResult[Double], Seq[Double])]): XYChart = {
    val chart = new XYChart(500, 500)

    val flattened: Iterable[(Double, Double, Double)] = data.flatMap {
      case (pred, actual: Seq[Double]) =>
        actual.lazyZip(pred.expected).lazyZip(pred.uncertainty().get.asInstanceOf[Seq[Double]]).toSeq
    }

    val actual = flattened.map(_._1).toArray
    val predicted = flattened.map(_._2).toArray
    val sigma = flattened.map(_._3).toArray

    chart.addSeries("data", actual, predicted, sigma)
    chart.getSeriesMap.get("data").setXYSeriesRenderStyle(XYSeriesRenderStyle.Scatter)

    val min = Math.min(actual.min, predicted.min)
    val max = Math.max(actual.max, predicted.max)
    chart.addSeries("ideal", Array(min, max), Array(min, max))

    chart.setTitle("Predicted vs actual")
    chart.setXAxisTitle("Actual")
    chart.setYAxisTitle("Predicted")

    chart
  }
}

/** Visualization of the error compared to the predicted uncertainty
  *
  * @param magnitude whether to plot the error or the magnitude (abs) of the error
  */
case class ErrorVsUncertainty(magnitude: Boolean = true) extends Visualization[Double] {

  override def visualize(data: Iterable[(PredictionResult[Double], Seq[Double])]): XYChart = {
    val chart = new XYChart(500, 500)

    val flattened: Iterable[(Double, Double)] = data.flatMap {
      case (pred, actual: Seq[Double]) =>
        val sigmas = pred.uncertainty().get.asInstanceOf[Seq[Double]]
        val errors = actual.zip(pred.expected).map {
          case (x, y) if magnitude => Math.abs(x - y)
          case (x, y)              => y - x
        }
        sigmas.zip(errors)
    }

    val sigma = flattened.map(_._1).toArray
    val error = flattened.map(_._2).toArray

    chart.addSeries("data", sigma, error)
    chart.getSeriesMap.get("data").setXYSeriesRenderStyle(XYSeriesRenderStyle.Scatter)

    val max = Math.max(sigma.max, error.map(Math.abs).max)
    if (magnitude) {
      chart.addSeries("ideal", Array(0.0, max), Array(0.0, max))
    } else {
      chart.addSeries("ideal", Array(max, 0.0, max), Array(-max, 0.0, max))
    }

    chart.setTitle("Error vs Uncertainty")
    chart.setXAxisTitle("Predicted uncertainty")
    if (magnitude) {
      chart.setYAxisTitle("Magnitude of error")
    } else {
      chart.setYAxisTitle("Error (predicted - actual)")
    }

    chart
  }
}
