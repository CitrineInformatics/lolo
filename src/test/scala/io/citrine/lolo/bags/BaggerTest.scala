package io.citrine.lolo.bags

import java.util.concurrent.{Callable, CancellationException, Executors, Future, TimeUnit}

import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.transformers.Standardizer
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.RegressionSplitter
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

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
    val DTLearner = RegressionTreeLearner(numFeatures = 3)
    val baggedLearner = Bagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(trainingData)
    val RF = RFMeta.getModel()

    assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
    assert(sigma.forall(_ > 0.0))

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
    val DTLearner = ClassificationTreeLearner()
    val baggedLearner = Bagger(DTLearner, numBags = trainingData.size / 2)
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
    * Test that the uncertainty metrics are properly calibrated
    *
    * This test is based on the "standard" RMSE, which is computed by dividing the error
    * by the predicted uncertainty.  On the exterior, these is an additional extrapolative
    * uncertainty that isn't captured well by this method, so we test the interior and the full
    * set independently
    */
  def testUncertaintyCalibration(): Unit = {
    val width = 0.10 // make the function more linear
    val nFeatures = 5
    val bagsPerRow = 4 // picked to be large enough that bias correction is small but model isn't too expensive
    val trainingData = TestUtils.generateTrainingData(128, nFeatures, xscale = width, seed = Random.nextLong())
    val DTLearner = RegressionTreeLearner(numFeatures = nFeatures)
    val bias = RegressionTreeLearner(maxDepth = 4)
    val baggedLearner = Bagger(DTLearner, numBags = bagsPerRow * trainingData.size) // c, biasLearner = Some(bias))
    val RFMeta = baggedLearner.train(trainingData)
    val RF = RFMeta.getModel()

    val interiorTestSet = TestUtils.generateTrainingData(128, nFeatures, xscale = width / 2.0, xoff = width / 4.0, seed = Random.nextLong())
    val fullTestSet = TestUtils.generateTrainingData(128, nFeatures, xscale = width, seed = Random.nextLong())

    val interiorStandardRMSE = BaggerTest.getStandardRMSE(interiorTestSet, RF)
    val fullStandardRMSE = BaggerTest.getStandardRMSE(fullTestSet, RF)
    assert(interiorStandardRMSE > 0.50, "Standard RMSE in the interior should be greater than 0.5")
    assert(interiorStandardRMSE < 1.50, "Standard RMSE in the interior should be less than 1.5")

    assert(fullStandardRMSE < 2.5, "Standard RMSE over the full domain should be less than 2.5")
    assert(fullStandardRMSE > 1.0, "Standard RMSE over the full domain should be greater than 1.0")
  }

  /**
    * Test the behavior of a random forest when the labels are constant
    */
  @Test
  def testUncertaintyCalibrationWithConstantResponse(): Unit = {
    // setup some training data with constant labels
    val nFeatures = 5
    val X: Vector[Vector[Any]] = TestUtils.generateTrainingData(128, nFeatures, xscale = 0.5, seed = Random.nextLong()).map(_._1)
    val y: Vector[Any] = X.map(_ => 0.0)

    // setup a relatively complicated random forest (turn a bunch of stuff on)
    val DTLearner = RegressionTreeLearner(
      numFeatures = nFeatures,
      leafLearner = Some(GuessTheMeanLearner()),
      maxDepth = 30,
      splitter = RegressionSplitter(randomizePivotLocation = true)
    )

    val bagger = new Bagger(
      new Standardizer(DTLearner),
      numBags = 64,
      useJackknife = true,
      /*
      biasLearner = Some(RegressionTreeLearner(
        maxDepth = 3,
        leafLearner = Some(GuessTheMeanLearner()),
        splitter = RegressionSplitter(randomizePivotLocation = true))
      ),
       */
      uncertaintyCalibration = true
    )

    // Make sure the model trains
    val model = bagger.train(X.zip(y)).getModel()

    // Generate a new test set and make sure the predictions are 0 +/- 0
    val testX: Vector[Vector[Any]] = TestUtils.generateTrainingData(128, nFeatures, xscale = 0.5, seed = Random.nextLong()).map(_._1)
    val predictions = model.transform(testX)
    assert(predictions.getExpected().forall(_ == 0.0))
    assert(predictions.getUncertainty().get.forall(_ == 0.0))
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
    val DTLearner = RegressionTreeLearner()
    val baggedLearner = Bagger(DTLearner, numBags = trainingData.size * 16) // use lots of trees to reduce noise
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
    * Test the relative training time of a calibrated vs an uncalibrated forest
    */
  @Test
  def calibrationTimeTest(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )
    val DTLearner = RegressionTreeLearner(numFeatures = 3)
    val start = System.nanoTime()
    Bagger(DTLearner, numBags = trainingData.size, uncertaintyCalibration = false)
      .train(trainingData)
      .getModel()
    val unCalibratedTime = 1.0e-9 * (System.nanoTime() - start)

    val startAgain = System.nanoTime()
    Bagger(DTLearner, numBags = 64, uncertaintyCalibration = true)
      .train(trainingData)
      .getModel()
    val calibratedTime = 1.0e-9 * (System.nanoTime() - startAgain)

    assert(calibratedTime < unCalibratedTime, s"The calibration scheme has experienced a dramatic slowdown")
  }

  /**
    * Test that the bagged learner can be interrupted
    */
  @Test
  def testInterrupt(): Unit = {
    val trainingData = TestUtils.generateTrainingData(2048, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val DTLearner = RegressionTreeLearner(numFeatures = 3)
    val baggedLearner = Bagger(DTLearner, numBags = trainingData.size)

    // Create a future to run train
    val tmpPool = Executors.newFixedThreadPool(1)
    val fut: Future[BaggedTrainingResult] = tmpPool.submit(
      new Callable[BaggedTrainingResult] {
        override def call() = {
          val res = baggedLearner.train(trainingData)
          assert(false, "Training was not terminated")
          res
        }
      }
    )
    // Let the thread start
    Thread.sleep(1000)

    // Cancel it
    val start = System.currentTimeMillis()
    assert(fut.cancel(true), "Failed to cancel future")

    // Make sure we get either a cancellation of interrupted exception
    try {
      fut.get()
      assert(false, "Future completed")
    } catch {
      case _: CancellationException =>
      case _: InterruptedException =>
      case _: Throwable => assert(false, "Future threw an exception")
    }

    // Shutdown the pool
    tmpPool.shutdown()
    assert(tmpPool.awaitTermination(1, TimeUnit.MINUTES), "Thread pool didn't terminate after a minute!")

    // Did it halt fast enough?
    val totalTime = (System.currentTimeMillis() - start) * 1.0e-3
    assert(totalTime < 2.0, "Thread took too long to terminate")
  }

  /**
    * Test that uncertainty recalibration functions correctly with small amounts of data. In some cases,
    * especially with a simple, binary function, it is possible to have every tree make the correct prediction, leading
    * to uncertainty = 0, and if not handled correctly, an uncertainty rescaling ratio that is NaN.
    *
    * Note that this is an unusual case, and is not caught by testUncertaintyFloor()
    */
  @Test
  def testSmallDataRecalibration(): Unit = {
    // Define a simple, binary function and create training data
    def stepFunction(x: Seq[Double]): Double = Math.floor(2 * x(0))

    val trainingData = TestUtils.generateTrainingData(rows = 16, cols = 2, function = stepFunction)

    /* Create a bagger out of GuessTheMean learners, and train the model.
     * This model has a rescale field, which should be a real number. If it is not,
     * then the model will fail to train
     */
    val DTLearner = RegressionTreeLearner(
      leafLearner = Some(GuessTheMeanLearner()),
      numFeatures = 2,
      splitter = RegressionSplitter(randomizePivotLocation = true)
    )
    val trainedModel: BaggedModel = Bagger(DTLearner,
      numBags = 16, useJackknife = true, uncertaintyCalibration = true)
      .train(trainingData)
      .getModel()
  }

  /**
    * Test that the uncertainty is always positive (and non-zero)
    *
    * This happens randomly, so let's repeat a test many times to make sure we catch it.  On my machine, this fails
    * in the first couple thousand times and takes runs for 13 seconds once its resolved, so I don't think
    * that's too much overhead.
    */
  @Test
  def testUncertaintyFloor(): Unit = {
    (0 until 16384).foreach { idx =>
      val trainingData = TestUtils.generateTrainingData(16, 5, noise = 0.0, function = Friedman.friedmanSilverman, seed = Random.nextLong())
      val DTLearner = RegressionTreeLearner(numFeatures = 2)
      val sigma = Bagger(DTLearner, numBags = 7)
        .train(trainingData)
        .getModel()
        .transform(trainingData.map(_._1))
        .getUncertainty().get.asInstanceOf[Seq[Double]]
      assert(sigma.forall(_ > 0.0), s"Found an predicted uncertainty of ${sigma.min} during trial $idx")
    }
  }

  /**
    * Test that the uncertainty is always positive (and non-zero) when a bias model is used
    *
    * Most of the logic is covered by testUncertaintyFloor, and this test is heavier, so we'll run fewer iterations
    */
  @Test
  def testUncertaintyFloorWithBias(): Unit = {
    (0 until 1024).foreach { idx =>
      val trainingData = TestUtils.generateTrainingData(16, 5, noise = 0.0, function = Friedman.friedmanSilverman, seed = Random.nextLong())
      val DTLearner = RegressionTreeLearner(numFeatures = 2)
      val sigma = Bagger(DTLearner, numBags = 7) // , biasLearner = Some(GuessTheMeanLearner()))
        .train(trainingData)
        .getModel()
        .transform(trainingData.map(_._1))
        .getUncertainty().get.asInstanceOf[Seq[Double]]
      assert(sigma.forall(_ > 0.0), s"Found an predicted uncertainty of ${sigma.min} during trial $idx")
    }
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
      .testUncertaintyCalibration()
  }


  def getStandardRMSE(testSet: Seq[(Vector[Any], Double)], model: BaggedModel): Double = {
    val predictions = model.transform(testSet.map(_._1))
    val pva = testSet.map(_._2).zip(
      predictions.getExpected().asInstanceOf[Seq[Double]].zip(
        predictions.getUncertainty().get.asInstanceOf[Seq[Double]]
      )
    )
    val standardError = pva.map { case (a: Double, (p: Double, u: Double)) =>
      Math.abs(a - p) / u
    }
    Math.sqrt(standardError.map(Math.pow(_, 2.0)).sum / testSet.size)
  }

}
