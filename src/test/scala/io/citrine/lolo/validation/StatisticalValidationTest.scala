package io.citrine.lolo.validation

import io.citrine.lolo.TestUtils
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

class StatisticalValidationTest {

  /**
    * Check that the generative validation is approximately consistent with k-fold cross-validation
    *
    * This test is not supposed to be statistically rigorous; just specific enough to detect bugs in the implementation
    */
  @Test
  def testCompareToKFolds(): Unit = {
    val learner = RandomForest()
    val dataSet = TestUtils.generateTrainingData(128, 8, Friedman.friedmanSilverman)
    val dataGenerator = TestUtils.iterateTrainingData(8, Friedman.friedmanSilverman)

    val metrics = Map("rmse" -> RootMeanSquareError)
    val (rmseFromCV, uncertaintyFromCV) = CrossValidation.kFoldCrossvalidation(dataSet, learner, metrics, k = 4, nTrial = 4)("rmse")

    val (rmseFromStats, uncertaintyFromStats) = Metric.estimateMetrics(
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
}
