package io.citrine.lolo.validation

import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import io.citrine.lolo.learners.RandomForestRegressor
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

class CrossValidationTest extends SeedRandomMixIn {

  /**
    * Test that CV results are consistent with out-of-bag estimates from the bagged learner
    *
    * Who cross-validates the cross-validator?
    */
  @Test
  def testCompareToOutOfBag(): Unit = {
    val learner = RandomForestRegressor()
    val data = DataGenerator.generate(128, 8, Friedman.friedmanSilverman, rng = rng).data

    val metric = RootMeanSquareError
    val (rmseFromCV, uncertainty) =
      CrossValidation.kFoldCrossvalidation(data, learner, Map("rmse" -> metric), k = 3)("rmse")

    val trainingResult = learner.train(data)
    val rmseFromPVA = Math.sqrt(
      trainingResult
        .getPredictedVsActual()
        .get
        .map {
          case (_, p: Double, a: Double) => Math.pow(p - a, 2.0)
        }
        .sum / trainingResult.getPredictedVsActual().get.size
    )

    // These have a false negative rate less than 1/100 at the time of original authorship
    assert(rmseFromPVA < rmseFromCV + uncertainty)
    assert((rmseFromPVA - rmseFromCV) / rmseFromPVA < 0.2)
  }
}
