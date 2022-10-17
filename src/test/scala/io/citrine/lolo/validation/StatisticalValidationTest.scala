package io.citrine.lolo.validation

import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import io.citrine.lolo.learners.RandomForestRegressor
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

class StatisticalValidationTest extends SeedRandomMixIn {

  @Test
  def testCompareToKFolds(): Unit = {
    val learner = RandomForestRegressor()
    val dataSet = DataGenerator.generate(128, 8, Friedman.friedmanSilverman, rng = rng).data
    val dataGenerator = DataGenerator.iterate(8, Friedman.friedmanSilverman, rng = rng)

    val metrics = Map("rmse" -> RootMeanSquareError)
    val (rmseFromCV, uncertaintyFromCV) =
      CrossValidation.kFoldCrossvalidation(dataSet, learner, metrics, k = 4, nTrial = 4)("rmse")

    val (rmseFromStats, uncertaintyFromStats) = Merit.estimateMerits(
      StatisticalValidation().generativeValidation(dataGenerator, learner, 96, 32, 16, rng = rng),
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
