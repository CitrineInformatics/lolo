package io.citrine.lolo.validation

import io.citrine.lolo.{Learner, PredictionResult}

import scala.util.Random

/**
  * Methods that draw data from a distribution and compute predicted-vs-actual data
  */
case class StatisticalValidation(rng: Random = Random) {

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
    * @tparam T type of the model
    * @return predicted-vs-actual data that can be fed into a metric or visualization
    */
  def generativeValidation[T](
                               source: Iterator[(Vector[Any], T)],
                               learner: Learner,
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int
                             ): Iterator[(PredictionResult[T], Seq[T])] = {
    Iterator.tabulate(nRound) { _ =>
      val trainingData: Seq[(Vector[Any], T)] = source.take(nTrain).toSeq
      val model = learner.train(trainingData).getModel()
      val testData: Seq[(Vector[Any], T)] = source.take(nTest).toSeq
      val predictions: PredictionResult[T] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[T]]
      (predictions, testData.map(_._2))
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
    * @tparam T type of the model
    * @return predicted-vs-actual data that can be fed into a metric or visualization
    */
  def generativeValidation[T](
                               source: Iterable[(Vector[Any], T)],
                               learner: Learner,
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int
                             ): Iterator[(PredictionResult[T], Seq[T])] = {
    Iterator.tabulate(nRound) { _ =>
      val subset = rng.shuffle(source).take(nTrain + nTest)
      val (trainingData: Seq[(Vector[Any], T)], testData: Seq[(Vector[Any], T)]) = subset.splitAt(nTrain)
      val model = learner.train(trainingData).getModel()
      val predictions: PredictionResult[T] = model.transform(testData.map(_._1)).asInstanceOf[PredictionResult[T]]
      (predictions, testData.map(_._2))
    }
  }

  def generativeValidationWithNoiseMulti(
                               source: Iterable[(Vector[Any], (Double, Double))],
                               learners: Seq[Learner],
                               nTrain: Int,
                               nTest: Int,
                               nRound: Int,
                               source2: Option[Iterable[(Vector[Any], (Double, Double))]] = None,
                               frequency: Option[Double] = None
                             ): Iterator[(Seq[PredictionResult[Double]], Seq[Double])] = {
    Iterator.tabulate(nRound) { idx =>
      println(s"Round ${idx} of ${nRound}")
      val (trainingData: Seq[(Vector[Any], (Double, Double))], testData: Seq[(Vector[Any], (Double, Double))]) = if (frequency.getOrElse(0.0) > 0.0) {
        val n2 = (nTrain * frequency.get).toInt
        val n1 = nTrain - n2
        val subset = rng.shuffle(source).take(nTest + n1)
        val (training1, test) = subset.splitAt(n1)
        val training = training1 ++ rng.shuffle(source2.get).take(n2).toVector
        (training, test)
      } else {
        val subset = rng.shuffle(source).take(nTrain + nTest)
        subset.splitAt(nTrain)
      }

      val noisedTraining = trainingData.map { case (f, (l, n)) =>
        (f, (l + rng.nextGaussian() * n, n))
      }
      val noisedTest = testData.map { case (f, (l, n)) =>
        (f, l + rng.nextGaussian() * n * 0)
      }

      val predictions: Seq[PredictionResult[Double]] = learners.map { learner =>
        val model = learner.train(noisedTraining).getModel()
        if (true) {
          val predictions = noisedTest.map { x =>
            model.transform(Seq(x._1)).asInstanceOf[PredictionResult[Double]]
          }
          new PredictionResult[Double] {
            override def getExpected(): Seq[Double] = predictions.map(_.getExpected().head)

            override def getUncertainty(observational: Boolean): Option[Seq[Any]] = {
              Some(predictions.map(_.getUncertainty(observational).get.head))
            }
          }
        } else {
          model.transform(noisedTest.map(_._1)).asInstanceOf[PredictionResult[Double]]
        }
      }
      (predictions, noisedTest.map(_._2))
    }
  }

  def generativeValidationWithNoise(
                                     source: Iterable[(Vector[Any], (Double, Double))],
                                     learner: Learner,
                                     nTrain: Int,
                                     nTest: Int,
                                     nRound: Int
                                   ): Iterator[(PredictionResult[Double], Seq[Double])] = {
    generativeValidationWithNoiseMulti(source, Seq(learner), nTrain, nTest, nRound).map{case (predictions, actual) =>
      (predictions.head, actual)
    }
  }
}
