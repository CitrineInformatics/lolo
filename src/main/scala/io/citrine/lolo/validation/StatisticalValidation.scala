package io.citrine.lolo.validation

import io.citrine.lolo.api.{Learner, PredictionResult, TrainingRow}
import io.citrine.random.Random

/**
  * Methods that draw data from a distribution and compute predicted-vs-actual data
  */
case class StatisticalValidation() {

  /**
    * Generate predicted-vs-actual data given a source of ground truth data and a learner
    *
    * Each predicted-vs-actual set (i.e. item in the returned iterable) comes from:
    *  - Drawing nTrain points from the source iterator
    *  - Training the learner on those nTrain points
    *  - Drawing nTest more points to form a test set
    *  - Applying the model to the test set inputs, and zipping with the test set ground truth responses
    * which is repeated nRound times
    *
    * @param source  of the training and test data
    * @param learner to validate
    * @param nTrain  size of each training set
    * @param nTest   size of each test set
    * @param nRound  number of train/test sets to draw and evaluate
    * @param rng     random number generator for reproducibility
    * @tparam T type of the model
    * @return predicted-vs-actual data that can be fed into a metric or visualization
    */
  def generativeValidation[T](
      source: Iterator[TrainingRow[T]],
      learner: Learner[T],
      nTrain: Int,
      nTest: Int,
      nRound: Int,
      rng: Random
  ): Iterator[(PredictionResult[T], Seq[T])] = {
    Iterator.tabulate(nRound) { _ =>
      val trainingData = source.take(nTrain).toSeq
      val model = learner.train(trainingData, rng = rng).model
      val testData = source.take(nTest).toSeq
      val predictions: PredictionResult[T] = model.transform(testData.map(_.inputs))
      (predictions, testData.map(_.label))
    }
  }

  /**
    * Generate predicted-vs-actual data given a source of ground truth data and a learner
    *
    * Each predicted-vs-actual set (i.e. item in the returned iterable) comes from:
    *  - Drawing nTrain points from the source iterator
    *  - Training the learner on those nTrain points
    *  - Drawing nTest more points to form a test set
    *  - Applying the model to the test set inputs, and zipping with the test set ground truth responses
    * which is repeated nRound times
    *
    * @param source  of the training and test data
    * @param learner to validate
    * @param nTrain  size of each training set
    * @param nTest   size of each test set
    * @param nRound  number of train/test sets to draw and evaluate
    * @param rng     random number generator for reproducibility
    * @tparam T type of the model
    * @return predicted-vs-actual data that can be fed into a metric or visualization
    */
  def generativeValidation[T](
      source: Iterable[TrainingRow[T]],
      learner: Learner[T],
      nTrain: Int,
      nTest: Int,
      nRound: Int,
      rng: Random
  ): Iterator[(PredictionResult[T], Seq[T])] = {
    Iterator.tabulate(nRound) { _ =>
      val subset = rng.shuffle(source).take(nTrain + nTest)
      val (trainingData, testData) = subset.toVector.splitAt(nTrain)
      val model = learner.train(trainingData, rng = rng).model
      val predictions: PredictionResult[T] = model.transform(testData.map(_.inputs))
      (predictions, testData.map(_.label))
    }
  }
}
