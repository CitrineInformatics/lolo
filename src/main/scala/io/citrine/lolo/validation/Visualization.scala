package io.citrine.lolo.validation

import io.citrine.lolo.PredictionResult
import org.apache.commons.math3.distribution.CauchyDistribution
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle
import org.knowm.xchart._
import org.knowm.xchart.internal.chartpart.Chart
import org.knowm.xchart.internal.series.Series

trait Visualization[T] {
  def visualize(data: Iterable[(PredictionResult[T], Seq[T])]): Chart[_, _]
}

case class StandardResidualHistogram(nBins: Int = 128, range: Double = 8.0) extends Visualization[Double] {

  override def visualize(data: Iterable[(PredictionResult[Double], Seq[Double])]): CategoryChart = {
    val pua: Seq[(Double, Double, Double)] = data.flatMap { case (predictions, actual) =>
      (predictions.getExpected(), predictions.getUncertainty().get.asInstanceOf[Seq[Double]], actual).zipped.toSeq
    }.toSeq

    val standardErrors = pua.map { case (predicted, sigma, actual) => (predicted - actual) / sigma }

    val bins = (-nBins / 2 until nBins / 2).map { idx =>
      (range * idx.toDouble / nBins, range * (idx.toDouble + 1) / nBins)
    }

    val counts = bins.map { case (min, max) =>
      ((min + max) / 2.0, standardErrors.count(x => x >= min && x < max) / (standardErrors.size * (max - min)))
    }

    val chart: CategoryChart = new CategoryChartBuilder().build()
    chart.addSeries("data", counts.map(_._1).toArray, counts.map(_._2.toDouble).toArray)
    val normalVar = 1.0
    val normalSeries = counts.map(_._1).map(x => Math.exp(-x * x / (2 * normalVar)) / Math.sqrt(2 * Math.PI * normalVar))
    chart.addSeries(f"sigma=${Math.sqrt(normalVar)}%6.3f", counts.map(_._1).toArray, normalSeries.toArray)
    val gamma = 1.0
    val cauchy1 = new CauchyDistribution(0.0, gamma)
    val cauchySeries = counts.map(_._1).map(x => cauchy1.density(x))
    chart.addSeries(f"gamma=${gamma}%6.3f", counts.map(_._1).toArray, cauchySeries.toArray)

    chart.setTitle("(predicted - actual) / (predicted uncertainty)")
    // chart.setXAxisLabelOverrideMap(Map[java.lang.Double, AnyRef]().asJava)
    chart.setYAxisTitle("probability density")

    chart
  }
}

case class PredictedVsActual() extends Visualization[Double] {

  override def visualize(data: Iterable[(PredictionResult[Double], Seq[Double])]): XYChart = {
    val chart = new XYChart(500, 500)

    val flattened: Iterable[(Double, Double, Double)] = data.flatMap{case (pred, actual: Seq[Double]) =>
      val foo: Seq[(Double, Double, Double)] = (actual, pred.getExpected(), pred.getUncertainty().get.asInstanceOf[Seq[Double]]).zipped.toSeq.toSeq.toSeq.toSeq
      foo
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

    // chart.getStyler.setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter)

    chart
  }
}

case class ErrorvsUncertainty() extends Visualization[Double] {

  override def visualize(data: Iterable[(PredictionResult[Double], Seq[Double])]): XYChart = {
    val chart = new XYChart(500, 500)

    val flattened: Iterable[(Double, Double)] = data.flatMap{case (pred, actual: Seq[Double]) =>
      (pred.getUncertainty().get.asInstanceOf[Seq[Double]], actual.zip(pred.getExpected()).map{case (x, y) => Math.abs(x - y)}).zipped.toSeq
    }

    val sigma = flattened.map(_._1).toArray
    val error = flattened.map(_._2).toArray


    chart.addSeries("data", sigma, error)
    chart.getSeriesMap.get("data").setXYSeriesRenderStyle(XYSeriesRenderStyle.Scatter)

    val max = Math.max(sigma.max, error.max)
    chart.addSeries("ideal", Array(0.0, max), Array(0.0, max))

    chart.setTitle("Error vs Uncertainty")
    chart.setXAxisTitle("Predicted uncertainty")
    chart.setYAxisTitle("Magnitude of error")

    // chart.getStyler.setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter)

    chart
  }
}
