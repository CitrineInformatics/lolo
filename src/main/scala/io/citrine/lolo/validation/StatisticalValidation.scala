package io.citrine.lolo.validation

import io.citrine.lolo.{Learner, PredictionResult}
import org.apache.commons.math3.distribution.{CauchyDistribution, MultivariateNormalDistribution}
import org.knowm.xchart.{CategoryChart, CategoryChartBuilder}

import scala.util.Random
import scala.collection.JavaConverters._

/**
  * Methods that draw data from a distribution and compute predicted-vs-actual data
  */
object StatisticalValidation {

  def generativeValidation[T](
                               source: Iterator[(Vector[Any], T)],
                               learner: Learner,
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int
                             ): Iterable[(PredictionResult[T], Seq[T])] = {
    (0 until nRound).map { _ =>
      val trainingData: Seq[(Vector[Any], T)] = source.take(nTrain).toSeq
      val model = learner.train(trainingData).getModel()
      val testData: Seq[(Vector[Any], T)] = source.take(nTest).toSeq
      val predictions: PredictionResult[T] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[T]]
      (predictions, testData.map(_._2))
    }
  }



}
