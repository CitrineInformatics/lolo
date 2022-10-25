package io.citrine.lolo.bags

import breeze.linalg.DenseMatrix
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TestUtils, TrainingRow}
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.transformers.rotator.FeatureRotator
import io.citrine.lolo.transformers.standardizer.RegressionStandardizer
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.RegressionSplitter
import org.junit.Test
import org.scalatest.Assertions._

import java.io.{File, PrintWriter}

@Test
class BaggerTest extends SeedRandomMixIn {

  /**
    * Test the fit performance of an ensemble of ridge regression learners
    */
  @Test
  def testLinearEnsemble(): Unit = {
    val beta0 = Vector(0.1, 5.0, 3.0, -2.0, 4.0)
    def linearFunction(x: Seq[Double], beta: Seq[Double]): Double = {
      x.zip(beta).map { case (xi, w) => xi * w }.sum
    }

    val trainingData = DataGenerator
      .generate(rows = 256, beta0.length, noise = 0.5, function = x => linearFunction(x, beta0), rng = rng)
      .data

    val baseLearner = RegressionStandardizer(LinearRegressionLearner(regParam = Some(0.5)))
    val baggedLearner = RegressionBagger(baseLearner, numBags = trainingData.size)

    val learnerMeta = baggedLearner.train(trainingData, rng = rng)
    val model = learnerMeta.getModel()

    assert(learnerMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

    val prediction = model.transform(trainingData.take(1).map(_.inputs))
    val sigma = prediction.getUncertainty().get.map(_.asInstanceOf[Double])
    assert(sigma.forall(_ > 0.0))
    assert(prediction.getGradient().isDefined, "No gradient returned for linear ensemble.")
  }

  /**
    * Test the fit performance of the regression bagger
    */
  @Test
  def testRegressionBagger(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    val DTLearner = RegressionTreeLearner(numFeatures = 3)
    val baggedLearner = RegressionBagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(trainingData, rng = rng)
    val RF = RFMeta.getModel()

    assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

    val results = RF.transform(trainingData.map(_.inputs))
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
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .withBinnedLabels(bins = 8)
      .data

    val DTLearner = ClassificationTreeLearner()
    val baggedLearner = ClassificationBagger(DTLearner, numBags = trainingData.size / 2)
    val RFMeta = baggedLearner.train(trainingData, rng = rng)
    val RF = RFMeta.getModel()

    /* Inspect the results */
    val results = RF.transform(trainingData.map(_.inputs))
    val means = results.getExpected()
    assert(trainingData.map(_.label).zip(means).forall { case (a, p) => a == p })

    val uncertainty = results.getUncertainty()
    assert(uncertainty.isDefined)
    assert(trainingData.map(_.label).zip(uncertainty.get).forall {
      case (a, probs) =>
        val maxProb = probs(a)
        maxProb > 0.5 && maxProb < 1.0 && Math.abs(probs.values.sum - 1.0) < 1.0e-6
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
    val trainingData = DataGenerator.generate(128, nFeatures, xscale = width, rng = rng).data
    val DTLearner = RegressionTreeLearner(numFeatures = nFeatures)
    val bias = RegressionTreeLearner(maxDepth = 4)
    val baggedLearner = RegressionBagger(DTLearner, numBags = bagsPerRow * trainingData.size, biasLearner = Some(bias))
    val RFMeta = baggedLearner.train(trainingData, rng = rng)
    val RF = RFMeta.getModel()

    val interiorTestSet = DataGenerator
      .generate(128, nFeatures, xscale = width / 2.0, xoff = width / 4.0, rng = rng)
      .data
      .map(row => (row.inputs, row.label))
    val fullTestSet =
      DataGenerator.generate(128, nFeatures, xscale = width, rng = rng).data.map(row => (row.inputs, row.label))

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
    val constantData = DataGenerator.generate(128, nFeatures, xscale = 0.5, rng = rng).data.map(_.withLabel(0.0))

    // setup a relatively complicated random forest (turn a bunch of stuff on)
    val DTLearner = RegressionTreeLearner(
      numFeatures = nFeatures,
      leafLearner = Some(GuessTheMeanLearner()),
      maxDepth = 30,
      splitter = RegressionSplitter(randomizePivotLocation = true)
    )

    val bagger = RegressionBagger(
      RegressionStandardizer(DTLearner),
      numBags = 64,
      useJackknife = true,
      biasLearner = Some(
        RegressionTreeLearner(
          maxDepth = 3,
          leafLearner = Some(GuessTheMeanLearner()),
          splitter = RegressionSplitter(randomizePivotLocation = true)
        )
      ),
      uncertaintyCalibration = true
    )

    // Make sure the model trains
    val model = bagger.train(constantData, rng = rng).getModel()

    // Generate a new test set and make sure the predictions are 0 +/- 0
    val testX = DataGenerator.generate(128, nFeatures, xscale = 0.5, rng = rng).data.map(_.inputs)
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
    val trainingData = csv.map(vec => TrainingRow(vec.init, vec.last.asInstanceOf[Double]))
    val DTLearner = RegressionTreeLearner()
    val baggedLearner = RegressionBagger(
      DTLearner,
      numBags = trainingData.size * 16
    ) // use lots of trees to reduce noise
    val RF = baggedLearner.train(trainingData, rng = rng).getModel()

    /* Call transform on the training data */
    val results = RF.transform(trainingData.map(_.inputs))
    val scores = results.getImportanceScores().get
    val corners = Seq(0, 7, 56, 63)
    corners.foreach { i =>
      assert(
        scores(i)(i) == scores(i).max,
        s"The corner at $i didn't have the highest score: ${scores(i)(i)} vs ${scores(i).max}"
      )
    }
  }

  /**
    * Test the relative training time of a calibrated vs an uncalibrated forest
    */
  @Test
  def calibrationTimeTest(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    val DTLearner = RegressionTreeLearner(numFeatures = 3)
    val start = System.nanoTime()
    RegressionBagger(
      DTLearner,
      numBags = trainingData.size,
      uncertaintyCalibration = false
    )
      .train(trainingData, rng = rng)
      .getModel()
    val unCalibratedTime = 1.0e-9 * (System.nanoTime() - start)

    val startAgain = System.nanoTime()
    RegressionBagger(
      DTLearner,
      numBags = 64,
      uncertaintyCalibration = true
    )
      .train(trainingData, rng = rng)
      .getModel()
    val calibratedTime = 1.0e-9 * (System.nanoTime() - startAgain)

    assert(calibratedTime < unCalibratedTime, s"The calibration scheme has experienced a dramatic slowdown")
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

    val trainingData = DataGenerator.generate(rows = 16, cols = 2, function = stepFunction, rng = rng).data

    /* Create a bagger out of GuessTheMean learners, and train the model.
     * This model has a rescale field, which should be a real number. If it is not,
     * then the model will fail to train
     */
    val DTLearner = RegressionTreeLearner(
      leafLearner = Some(GuessTheMeanLearner()),
      numFeatures = 2,
      splitter = RegressionSplitter(randomizePivotLocation = true)
    )
    val trainedModel = RegressionBagger(
      DTLearner,
      numBags = 16,
      useJackknife = true,
      uncertaintyCalibration = true
    )
      .train(trainingData, rng = rng)
      .getModel()

    try {
      val _: BaggedModel[Any] = RegressionBagger(
        DTLearner,
        numBags = 16,
        useJackknife = true,
        uncertaintyCalibration = true,
        disableBootstrap = true
      )
        .train(trainingData, rng = rng)
        .getModel()
      fail("Setting both uncertaintyCalibration and disableBootstrap should throw an exception.")
    } catch {
      case _: Throwable =>
    }
  }

  /**
    * Test that the uncertainty is always positive (and non-zero)
    *
    * This happens randomly, so let's repeat a test many times to make sure we catch it.  On my machine, this fails
    * in the first couple thousand times and takes runs for 13 seconds once it's resolved, so I don't think
    * that's too much overhead.
    */
  @Test
  def testUncertaintyFloor(): Unit = {
    (0 until 16384).foreach { idx =>
      val trainingData =
        DataGenerator.generate(16, 5, noise = 0.0, function = Friedman.friedmanSilverman, rng = rng).data
      val DTLearner = RegressionTreeLearner(numFeatures = 2)
      val sigma = RegressionBagger(DTLearner, numBags = 7)
        .train(trainingData, rng = rng)
        .getModel()
        .transform(trainingData.map(_.inputs))
        .getUncertainty()
        .get
        .asInstanceOf[Seq[Double]]
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
      val trainingData =
        DataGenerator.generate(16, 5, noise = 0.0, function = Friedman.friedmanSilverman, rng = rng).data
      val DTLearner = RegressionTreeLearner(numFeatures = 2)
      val sigma = RegressionBagger(
        DTLearner,
        numBags = 7,
        biasLearner = Some(GuessTheMeanLearner())
      )
        .train(trainingData, rng = rng)
        .getModel()
        .transform(trainingData.map(_.inputs))
        .getUncertainty()
        .get
        .asInstanceOf[Seq[Double]]
      assert(sigma.forall(_ > 0.0), s"Found an predicted uncertainty of ${sigma.min} during trial $idx")
    }
  }

  /**
    * Test Shapley values are correctly averaged over trees.
    */
  @Test
  def testShapley(): Unit = {
    val nCols = 5
    val trainingData =
      DataGenerator.generate(64, nCols, noise = 0.0, function = Friedman.friedmanSilverman, rng = rng).data
    val DTLearner = RegressionTreeLearner(numFeatures = nCols)
    val model = RegressionBagger(DTLearner).train(trainingData, rng = rng).getModel()
    val trees = model.ensembleModels
    trainingData.foreach {
      case TrainingRow(x, _, _) =>
        val shapley = model.shapley(x).get

        // Do a quick sanity check on the output format.
        assert(shapley.cols == nCols)
        assert(shapley.rows == 1)

        // Compute the mean shap value over trees and ensure the bagged model gives the same result.
        val treeMean = (1.0 / trees.length) * trees
          .map { t =>
            t.shapley(x).get
          }
          .reduce[DenseMatrix[Double]] {
            case (a: DenseMatrix[Double], b: DenseMatrix[Double]) =>
              a +:+ b
          }
        val atol = 1e-8
        assert(
          (treeMean - shapley).toDenseVector.toScalaVector.forall { x => Math.abs(x) < atol }
        )
    }
  }

  /**
    * Test Shapley is None if ensemble doesn't return values
    */
  @Test
  def testShapleyIsEmpty(): Unit = {
    val nCols = 2
    val trainingData =
      DataGenerator.generate(8, nCols, noise = 0.0, function = Friedman.friedmanSilverman, rng = rng).data
    val learner = FeatureRotator(RegressionTreeLearner(numFeatures = nCols))
    val model = RegressionBagger(learner)
      .train(trainingData, rng = rng)
      .getModel()

    val x = trainingData.head.inputs
    assert(model.shapley(x).isEmpty)
  }
}

/**
  * Companion driver
  */
object BaggerTest extends SeedRandomMixIn {

  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    measureShapleyPerf()
  }

  /**
    * Simple driver for running a performance test of BaggedModel.shapley().
    */
  def measureShapleyPerf(): Unit = {
    val pw = new PrintWriter(new File(s"/tmp/shapley-perf_${rng.nextInt()}.tsv"))
    pw.write("nCols\tnRows\trepNum\trowIdx\tns\n")
    (256 to 1024 by 256).foreach { nCols =>
      (4 to 8 by 2).foreach { nRowsLog: Int =>
        val nRows = 1 << nRowsLog
        (1 to 3).foreach { repNum =>
          val trainingData =
            DataGenerator.generate(nRows, nCols, noise = 0.0, function = Friedman.friedmanSilverman, rng = rng).data
          val DTLearner = RegressionTreeLearner(numFeatures = nCols)
          println(s"Training model nCols=$nCols\tnRows=$nRows\trepNum=$repNum")
          val model = RegressionBagger(DTLearner)
            .train(trainingData, rng = rng)
            .getModel()
          println(s"Trained")

          rng.shuffle(trainingData).take(16).zipWithIndex.foreach {
            case (x, i) =>
              val t0 = System.nanoTime()
              val shapley = model.shapley(x.inputs).get
              val t1 = System.nanoTime()
              pw.write(s"$nCols\t$nRows\t$repNum\t$i\t${t1 - t0}\n")
              pw.flush()
          }
        }
      }
    }
    pw.close()
  }

  def getStandardRMSE(testSet: Seq[(Vector[Any], Double)], model: BaggedModel[Any]): Double = {
    val predictions = model.transform(testSet.map(_._1))
    val pva = testSet
      .map(_._2)
      .zip(
        predictions
          .getExpected()
          .asInstanceOf[Seq[Double]]
          .zip(
            predictions.getUncertainty().get.asInstanceOf[Seq[Double]]
          )
      )
    val standardError = pva.map {
      case (a: Double, (p: Double, u: Double)) =>
        Math.abs(a - p) / u
    }
    Math.sqrt(standardError.map(Math.pow(_, 2.0)).sum / testSet.size)
  }
}
