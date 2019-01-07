package io.citrine.lolo.validation

import io.citrine.lolo.{Learner, PredictionResult}
import org.apache.commons.math3.distribution.{CauchyDistribution, TDistribution}
import org.knowm.xchart.{CategoryChart, CategoryChartBuilder}
import org.knowm.xchart.internal.chartpart.Chart

object StatisticalValidation {

  def generativeValidation[T](
                               source: Iterator[(Vector[Any], T)],
                               learner: Learner,
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int,
                               metrics: Map[String, Metric[T]]
                             ): Map[String, (Double, Double)] = {
    (0 until nRound).flatMap{_ =>
      val trainingData: Seq[(Vector[Any], T)] = source.take(nTrain).toSeq
      val model = learner.train(trainingData).getModel()
      val testData: Seq[(Vector[Any], T)] = source.take(nTest).toSeq
      val predictions: PredictionResult[T] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[T]]
      metrics.mapValues(f => f.evaluate(predictions, testData.map(_._2))).toSeq
    }.groupBy(_._1).mapValues{x =>
      val metricResults = x.map(_._2)
      val mean = metricResults.sum / metricResults.size
      val variance = metricResults.map(y => Math.pow(y - mean, 2)).sum / metricResults.size
      (mean, Math.sqrt(variance / metricResults.size))
    }
  }

  def generativeHistogram(
                               source: Iterator[(Vector[Any], Double)],
                               learner: Learner,
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int
                             ): CategoryChart = {
    val errors: Seq[Double] = (0 until nRound).flatMap{_ =>
      val trainingData: Seq[(Vector[Any], Double)] = source.take(nTrain).toSeq
      val model = learner.train(trainingData).getModel()
      val testData: Seq[(Vector[Any], Double)] = source.take(nTest).toSeq
      val predictions: PredictionResult[Double] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[Double]]
      (predictions.getExpected(), predictions.getUncertainty().get, testData.map(_._2)).zipped.map{
        case (predicted: Double, sigma: Double, actual: Double) =>
          (predicted - actual) / sigma
      }
    }

    val range = Math.ceil(errors.map(Math.abs).max).toInt.toDouble
    val nBins = 100
    val bins = (-nBins/2 until nBins/2).map{idx =>
      (range * idx.toDouble / nBins, range * (idx.toDouble + 1) / nBins)
    }

    val counts = bins.map{case (min, max) =>
      ((min + max) / 2.0, errors.count(x => x >= min && x < max) / (errors.size * (max - min)))
    }

    val chart: CategoryChart = new CategoryChartBuilder().build()
    chart.addSeries("data", counts.map(_._1).toArray, counts.map(_._2.toDouble).toArray)
    chart.addSeries("normal", counts.map(_._1).toArray, counts.map(_._1).map(x => Math.exp(-x*x / 2)/Math.sqrt(2 * Math.PI)).toArray)
    // val tdist = new TDistribution(1)
    //c hart.addSeries("t=1", counts.map(_._1).toArray, counts.map(_._1).map(x => tdist.density(x)).toArray)
    val cauchy = new CauchyDistribution(0.0, 1.0/3.0)
    chart.addSeries("gamma=1", counts.map(_._1).toArray, counts.map(_._1).map(x => cauchy.density(x)).toArray)

    chart
  }

}
