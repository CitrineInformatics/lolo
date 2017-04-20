package io.citrine.lolo.bags

import java.util.concurrent.{Callable, CancellationException, Executors, Future}

import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class BaggerTest {

  /**
    * Test the fit performance of the regression bagger
    */
  @Test
  def testRegressionBagger(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )
    val DTLearner = new RegressionTreeLearner(numFeatures = 3)
    val baggedLearner = new Bagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(trainingData)
    val RF = RFMeta.getModel()

    assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
    assert(sigma.forall(_ >= 0.0))

    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")

    /* The first feature should be the most important */
    val importances = RFMeta.getFeatureImportance().get
    assert(importances(1) == importances.max)
  }

  /**
    * Test the fit performance of the classification bagger
    */
  @Test
  def testClassificationBagger(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )
    val DTLearner = new ClassificationTreeLearner()
    val baggedLearner = new Bagger(DTLearner, numBags = trainingData.size / 2)
    val RFMeta = baggedLearner.train(trainingData)
    val RF = RFMeta.getModel()

    /* Inspect the results */
    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    assert(trainingData.map(_._2).zip(means).forall { case (a, p) => a == p })

    val uncertainty = results.getUncertainty()
    assert(uncertainty.isDefined)
    assert(trainingData.map(_._2).zip(uncertainty.get).forall { case (a, probs) =>
      val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
      val maxProb = classProbabilities(a)
      maxProb > 0.5 && maxProb < 1.0 && Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6
    })
    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")

    /* The first feature should be the most important */
    val importances = RFMeta.getFeatureImportance().get
    assert(importances.slice(0, 5).min > importances.slice(5, importances.size).max)
  }

  /**
    * Test the scores on a smaller example, because computing them all can be expensive.
    *
    * In general, we don't even know that the self-score (score on a prediction on oneself) is maximal.  For example,
    * consider a training point that is sandwiched between two other points, i.e. y in | x     x y x    x |.  However,
    * this training data is on a 2D grid, so we know the corners of that grid need to have maximal self-scores.  Those
    * are at indices 0, 7, 56, and 63.
    */
  @Test
  def testScores(): Unit = {
    val csv = TestUtils.readCsv("double_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
    val DTLearner = new RegressionTreeLearner()
    val baggedLearner = new Bagger(DTLearner, numBags = trainingData.size * 16) // use lots of trees to reduce noise
    val RF = baggedLearner.train(trainingData).getModel()

    /* Call transform on the training data */
    val results = RF.transform(trainingData.map(_._1))
    val scores = results.getImportanceScores().get
    val corners = Seq(0, 7, 56, 63)
    assert(
      corners.forall(i => scores(i)(i) == scores(i).max),
      "One of the training corners didn't have the highest score"
    )
  }

  /**
    * Test that the bagged learner can be interrupted
    */
  @Test
  def testInterrupt(): Unit = {
    val trainingData = TestUtils.generateTrainingData(2048, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val DTLearner = new RegressionTreeLearner(numFeatures = 3)
    val baggedLearner = new Bagger(DTLearner, numBags = trainingData.size)

    val tmpPool = Executors.newFixedThreadPool(1)
    val fut: Future[BaggedTrainingResult] = tmpPool.submit(
      new Callable[BaggedTrainingResult] {
        override def call() = {
          println("Starting to run the future")
          val res = baggedLearner.train(trainingData)
          assert(false, "Training was not terminated")
          res
        }
      }
    )
    assert(fut.cancel(true), "Failed to cancel future")

    try {
      fut.get()
      assert(false, "Future completed")
    } catch {
      case _: CancellationException =>
      case _: InterruptedException =>
      case _: Throwable => assert(false)
    }
    tmpPool.shutdown()
  }
}

/**
  * Companion driver
  */
object BaggerTest {
  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    new BaggerTest()
      .testInterupt()
  }
}
