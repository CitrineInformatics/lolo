package io.citrine.lolo.validation

import io.citrine.lolo.TestUtils
import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.transformers.Standardizer
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

class StatisticalValidationTest {

  @Test
  def testCompareToKFolds(): Unit = {
    val learner = RandomForest()
    val dataSet = TestUtils.generateTrainingData(128, 8, Friedman.friedmanSilverman)
    val dataGenerator = TestUtils.iterateTrainingData(8, Friedman.friedmanSilverman)

    val metrics = Map("rmse" -> RootMeanSquareError)
    val (rmseFromCV, uncertaintyFromCV) = CrossValidation.kFoldCrossvalidation(dataSet, learner, metrics, k = 4, nTrial = 4)("rmse")

    val (rmseFromStats, uncertaintyFromStats) = Merit.estimateMerits(
      StatisticalValidation.generativeValidation(dataGenerator, learner, 96, 32, 16),
      metrics
    )("rmse")

    // These assertions have a false negative rate of 1/1000
    assert(
      uncertaintyFromCV + uncertaintyFromStats < (rmseFromCV + rmseFromStats) / 4.0,
      s"The uncertainty is more than half the metric value; that's too uncertain to test"
    )
    assert(
      Math.abs(rmseFromCV - rmseFromStats) < 2.0 * (uncertaintyFromCV + uncertaintyFromStats),
      s"The CV and statistical validation methods disagree more than 2 sigma-sum"
    )
  }

  @Test
  def testUncertaintyStability(): Unit = {
    val trainingData: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(256, 10, xscale = 100, function = { x =>
      x.head * 2.0
    })

    val featureSize = 13
    val leafLearner = Some(GuessTheMeanLearner())
    val maxDepth = 30
    val numBags = 64
    val useJacknife = false
    val biasModelDepth = 5
    val recalibrate = false

    val DTLearner = new RegressionTreeLearner(
      numFeatures = Math.max(featureSize, 1),
      leafLearner = leafLearner,
      maxDepth = maxDepth
    )

    val learner = new Bagger(
      new Standardizer(DTLearner),
      numBags = numBags,
      useJackknife = useJacknife,
      biasLearner = Some(new RegressionTreeLearner(
        maxDepth = biasModelDepth,
        leafLearner = Some(new GuessTheMeanLearner()))
      ),
      uncertaintyCalibration = recalibrate
    )

    val model = learner.train(trainingData.tail)

    val predictionCandidate: Vector[Double] = trainingData.head._1
    val seqResult = model.model.transform(Seq(predictionCandidate, predictionCandidate))
    val singleResult = model.model.transform(Seq(predictionCandidate))

    // Make sure we get the same results from the same candidate when
    // requesting a single candidate at a time as when predicting multiple
    assert(
      math.abs(singleResult.getUncertainty().get.head.asInstanceOf[Double] -
        seqResult.getUncertainty().get.head.asInstanceOf[Double]) < 1E-8
    )
  }
}
