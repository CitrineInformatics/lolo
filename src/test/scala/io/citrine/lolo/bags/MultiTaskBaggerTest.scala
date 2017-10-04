package io.citrine.lolo.bags

import java.util.concurrent.{Callable, CancellationException, Executors, Future, TimeUnit}

import io.citrine.lolo.TestUtils
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
    * Test the fit performance of the regression bagger
    */
  @Test
  def testSingleRegression(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(512, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val DTLearner = new MultiTaskTreeLearner()
    val baggedLearner = new MultiTaskBagger(DTLearner, numBags = trainingData.size)
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
    * Test the fit performance of the classification bagger
    */
  @Test
  def testClassificationBagger(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(128, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )
    val inputs = trainingData.map(_._1)
    val labels = trainingData.map(_._2)
    val DTLearner = new MultiTaskTreeLearner()
    val baggedLearner = new MultiTaskBagger(DTLearner, numBags = trainingData.size)
    val RFMeta = baggedLearner.train(inputs, Seq(labels)).head
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
  }


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
    val DTLearner = new MultiTaskTreeLearner()
    val baggedLearner = new MultiTaskBagger(DTLearner, numBags = inputs.size)
    val RFMeta = baggedLearner.train(inputs, Seq(realLabel, sparseCat)).last
    val RF = RFMeta.getModel()

    val catResults = RF.transform(inputs).getExpected()

    val reference = new Bagger(new ClassificationTreeLearner(), numBags = inputs.size)
      .train(inputs.zip(sparseCat).filterNot(_._2 == null))
      .getModel()
      .transform(inputs)
      .getExpected()

    val singleF1 = ClassificationMetrics.f1scores(reference, catLabel)
    val multiF1 = ClassificationMetrics.f1scores(catResults, catLabel)

    assert(multiF1 > singleF1)
    assert(multiF1 < 1.0)
  }
}

