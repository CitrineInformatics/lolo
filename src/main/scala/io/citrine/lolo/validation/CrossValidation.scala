package io.citrine.lolo.validation

import io.citrine.lolo.{Learner, PredictionResult}

import scala.util.Random

/**
  * Methods tha use cross-validation to calculate predicted-vs-actual data and metric estimates
  */
object CrossValidation {

  /**
    * Driver to apply named metrics to k-fold cross-validated predicted-vs-actual
    *
    * @param trainingData to cross-validate with
    * @param learner      to cross-validate
    * @param metrics      apply to the predicted-vs-actual data
    * @param k            number of folds
    * @param nTrial       number of times to refold the data to improve statistics
    * @tparam T type of the response, e.g. Double for scalar regression
    * @return a Map from the name of the metric to its mean value and the error in that mean
    */
  def kFoldCrossvalidation[T](
                               trainingData: Seq[(Vector[Any], T)],
                               learner: Learner,
                               metrics: Map[String, Merit[T]],
                               k: Int = 8,
                               nTrial: Int = 1
                             ): Map[String, (Double, Double)] = {
    Merit.estimateMerits(
      kFoldPvA(trainingData, learner, k, nTrial).iterator,
      metrics
    )
  }

  /**
    * Use k-fold cross-validation to create predicted vs actual results
    *
    * @param trainingData to cross-validate with
    * @param learner      to cross-validate
    * @param k            number of folds
    * @param nTrial       number of times to re-fold the data to improve statistics
    * @tparam T type of the response, e.g. Double for scalar regression
    * @return an iterable over predicted-vs-actual for each fold
    */
  def kFoldPvA[T](
                   trainingData: Seq[(Vector[Any], T)],
                   learner: Learner,
                   k: Int = 8,
                   nTrial: Int = 1
                 ): Iterable[(PredictionResult[T], Seq[T])] = {
    val nTest: Int = Math.ceil(trainingData.size.toDouble / k).toInt
    (0 until nTrial).flatMap { _ =>
      val folds: Seq[Seq[(Vector[Any], T)]] = Random.shuffle(trainingData).grouped(nTest).toSeq

      folds.indices.map { idx =>
        val (testFolds, trainFolds) = folds.zipWithIndex.partition(_._2 == idx)
        val testData = testFolds.flatMap(_._1)
        val trainData = trainFolds.flatMap(_._1)
        val model = learner.train(trainData).getModel()
        val predictions: PredictionResult[T] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[T]]
        (predictions, testData.map(_._2))
      }
    }
  }

}
