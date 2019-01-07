package io.citrine.lolo.validation

import io.citrine.lolo.{Learner, PredictionResult}

import scala.util.Random

object CrossValidation {

  def kFoldCrossvalidation[T](
                               trainingData: Seq[(Vector[Any], T)],
                               learner: Learner,
                               metrics: Map[String, Metric[T]],
                               k: Int = 8,
                               nTrial: Int = 1
                             ): Map[String, (Double, Double)] = {
    val nTest: Int = Math.ceil(trainingData.size.toDouble / k).toInt
    (0 until nTrial).flatMap{_ =>
      val folds: Seq[Seq[(Vector[Any], T)]] = Random.shuffle(trainingData).grouped(nTest).toSeq

      folds.indices.flatMap{ idx =>
        val (testFolds, trainFolds) = folds.zipWithIndex.partition(_._2 == idx)
        val testData = testFolds.flatMap(_._1)
        val trainData = trainFolds.flatMap(_._1)
        val model = learner.train(trainData).getModel()
        val predictions: PredictionResult[T] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[T]]
        metrics.mapValues(f => f.evaluate(predictions, testData.map(_._2))).toSeq
      }
    }.groupBy(_._1).mapValues{x =>
      val metricResults = x.map(_._2)
      val mean = metricResults.sum / metricResults.size
      val variance = metricResults.map(y => Math.pow(y - mean, 2)).sum / metricResults.size
      (mean, Math.sqrt(variance / metricResults.size))
    }
  }

}
