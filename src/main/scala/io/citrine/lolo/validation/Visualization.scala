package io.citrine.lolo.validation

import io.citrine.lolo.PredictionResult
import org.apache.commons.math3.distribution.CauchyDistribution
import org.knowm.xchart.{CategoryChart, CategoryChartBuilder, CategorySeries}
import org.knowm.xchart.internal.chartpart.Chart

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
