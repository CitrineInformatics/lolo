package io.citrine.lolo.validation

import io.citrine.lolo.PredictionResult
import org.knowm.xchart.BitmapEncoder.BitmapFormat
import org.knowm.xchart.{BitmapEncoder, CategoryChart, CategoryChartBuilder, SwingWrapper}

trait Metric[T] {
  def evaluate(predictionResult: PredictionResult[T], actual: Seq[T]): Double
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

     val range = 10.0
     val nBins = 100
     val bins = (-nBins/2 until nBins/2).map{idx =>
       (range * idx.toDouble / nBins, range * (idx.toDouble + 1) / nBins)
     }

     val counts = bins.map{case (min, max) =>
       ((min + max) / 2.0, standardized.count(x => x >= min && x < max))
     }

     val chart: CategoryChart = new CategoryChartBuilder().build()
     chart.addSeries("foo", counts.map(_._1).toArray, counts.map(_._2.toDouble).toArray)
     new SwingWrapper(chart).displayChart()
     BitmapEncoder.saveBitmap(chart, "./test_plot", BitmapFormat.PNG)

     Math.sqrt(standardized.map(Math.pow(_,2.0)).sum / standardized.size)
   }
}