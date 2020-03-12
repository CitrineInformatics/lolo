package io.citrine.lolo.bags

import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class MultiTaskBaggerTest {

  /**
    * Test that we get a reasonable output on a single regression problem
    */
  @Test
  def testSingleRegression(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(512, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(inputs, Seq(labels)).head
    val RF = RFMeta.getModel()


    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
    assert(sigma.forall(_ >= 0.0))

    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")
    assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")
  }

  /**
    * Test UQ on multitask regression.
    */
  @Test
  def testBaggedMultiTaskGetUncertainty(): Unit = {
    val noiseLevel = 100.0
    val rng = new Random(237485L)
    Seq(MultiTaskTreeLearner()).foreach{ baseLearner =>
      Seq(30,100,301).foreach { nRows =>
        val trainingDataTmp = TestUtils.generateTrainingData(nRows, 1, noise = 0.0, function = _ => 0.0, seed = rng.nextLong())
        val trainingData = trainingDataTmp.map { x => (x._1, x._2 + noiseLevel * rng.nextDouble()) }
        val inputs = trainingData.map(_._1)
        val labels = trainingData.map(_._2)
        val baggedLearner = MultiTaskBagger(baseLearner, numBags = 2 * nRows, uncertaintyCalibration = true)
        val RFMeta = baggedLearner.train(inputs, Seq(labels)).head
        val RF = RFMeta.getModel()

        val results = RF.transform(trainingData.take(1).map(_._1))
        val sigmaObs: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
        val sigmaMean: Seq[Double] = results.getUncertainty(observational = false).get.asInstanceOf[Seq[Double]]
// TODO(grobinson): enable this interface.
//        sigmaMean.zip(results.asInstanceOf[RegressionResult].getStdDevMean().get).foreach{ case (a,b) =>
//          assert(a == b, "Expected getUncertainty(observational=false)=getStdDevMean()")
//        }
//        sigmaObs.zip(results.asInstanceOf[RegressionResult].getStdDevObs().get).foreach{ case (a,b) =>
//          assert(a == b, "Expected getUncertainty()=getStdDevObs()")
//        }
//        sigmaObs.zip(sigmaMean).foreach { case (sObs, sMean) => assert(sObs > sMean, "Uncertainty should be greater when observational = true.") }

        if (baseLearner.isInstanceOf[GuessTheMeanLearner]) {
          val rtolLower = 5.0  // Future recalibration should decrease this number.
          val rtolUpper = 1.0  // Future recalibration should increase this number.
          sigmaObs.foreach { s =>
            assert(rtolLower * s > noiseLevel, "Observational StdDev getUncertainty() is too small.")
            assert(s < rtolUpper * noiseLevel, "Observational StdDev getUncertainty() is too large.")
          }
          sigmaMean.foreach { s =>
            assert(rtolLower * s > noiseLevel / Math.sqrt(nRows - 1), "Mean StdDev getUncertainty(observational=false) is too small.")
            assert(s < rtolUpper * noiseLevel / Math.sqrt(nRows - 1), "Mean StdDev getUncertainty(observational=false) is too large.")
          }
        }
      }
    }
  }

  /**
    * Test the we get a reasonable result on a single classification problem
    */
  @Test
  def testClassificationBagger(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(128, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(inputs, Seq(labels)).head
    val RF = RFMeta.getModel()

    /* Inspect the results */
    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    assert(trainingData.map(_._2).zip(means).forall { case (a, p) => a == p })

    val uncertainty = results.getUncertainty()
    assert(uncertainty.isDefined)
    trainingData.map(_._2).zip(uncertainty.get).foreach { case (a, probs) =>
      val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
      val maxProb = classProbabilities(a)
      assert(maxProb >= 0.5)
      assert(maxProb < 1.0)
      assert(Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6)
    }
    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")
  }

  /**
    * Test that multi-task with dense labels works, and remembers all its inputs
    */
  @Test
  def testMixed(): Unit = {
    /* Setup some data */
    val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(256, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val inputs: Seq[Vector[Double]] = raw.map(_._1)
    val realLabel: Seq[Double] = raw.map(_._2)
    val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)
    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = inputs.size, biasLearner = Some(new RegressionTreeLearner(maxDepth = 2)))
    val RFMeta = baggedLearner.train(inputs, Seq(realLabel, catLabel)).last
    val RF = RFMeta.getModel()

    val catResults = RF.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]
    assert(catResults.zip(catLabel).forall(p => p._1 == p._2))
  }

  /**
    * Test that multi-task (regression, classification) with sparse classification labels
    * outperforms a direct classification model on the same label
    */
  @Test
  def testSparseMixedBagged(): Unit = {
    /* Setup some data */
    val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(256, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val inputs: Seq[Vector[Double]] = raw.map(_._1)
    val realLabel: Seq[Double] = raw.map(_._2)
    val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)
    val sparseCat = catLabel.map(x =>
      if (Random.nextDouble() > 0.125) {
        null
      } else {
        x
      }
    )
    val sparseReal = realLabel.map(x =>
      if (Random.nextDouble() > 0.5) {
        Double.NaN
      } else {
        x
      }
    )

    val DTLearner = MultiTaskTreeLearner()
    val baggedLearner = MultiTaskBagger(DTLearner, numBags = inputs.size, biasLearner = Some(new GuessTheMeanLearner))
    val trainingResult = baggedLearner.train(inputs, Seq(sparseReal, sparseCat))
    val RFMeta = trainingResult.last
    val RF = RFMeta.getModel()

    val catResults = RF.transform(inputs).getExpected()
    val realUncertainty = trainingResult.head.getModel().transform(inputs).getUncertainty().get
    assert(realUncertainty.forall(!_.asInstanceOf[Double].isNaN), s"Some uncertainty values were NaN")

    val referenceModel = Bagger(ClassificationTreeLearner(), numBags = inputs.size)
      .train(inputs.zip(sparseCat).filterNot(_._2 == null))
    val reference = referenceModel
      .getModel()
      .transform(inputs)
      .getExpected()

    val singleF1 = ClassificationMetrics.f1scores(reference, catLabel)
    val multiF1 = ClassificationMetrics.f1scores(catResults, catLabel)

    // Make sure we can grab the loss without issue
    val singleLoss = referenceModel.getLoss().get
    val multiLoss = RFMeta.getLoss().get
    val regressionLoss = trainingResult.head.getLoss().get
    assert(!singleLoss.isNaN, "Single task classification loss was NaN")
    assert(!multiLoss.isNaN, "Sparse multitask classification loss was NaN")
    assert(!regressionLoss.isNaN, "Sparse regression loss was NaN")

    assert(multiF1 > singleF1, s"Multi-task is under-performing single-task")
    assert(multiF1 <= 1.0, "Multitask classification F1 score was greater than 1.0")
  }
}

