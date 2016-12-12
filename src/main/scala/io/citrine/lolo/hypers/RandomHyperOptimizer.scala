package io.citrine.lolo.hypers

import io.citrine.lolo.results.{TrainingResult, hasLoss}
import io.citrine.lolo.{Learner}

import scala.util.Random

/**
  * Search for hypers by randomly sampling the search space
  *
  * This optimizer can be evaluated multiple times and will persist the best results across those
  * calls.
  * Created by maxhutch on 12/7/16.
  *
  * @param base learner to optimize parameters for
  */
class RandomHyperOptimizer(base: Learner) extends HyperOptimizer(base) {

  /** Keep track of the best hypers outside of the optimize call so it persists across calls */
  var best: Map[String, Any] = Map()
  /** Likewise with the lowest loss */
  var loss = Double.MaxValue

  /**
    * Search over numIterations random draws for the search space
    *
    * @param trainingData  the data to train/test on
    * @param numIterations number of draws to take
    * @return the best hyper map found in give iterations and the corresponding loss
    */
  override def optimize(trainingData: Seq[(Vector[Any], Any)], numIterations: Int): (Map[String, Any], Double) = {
    /* Just draw numIteration times */
    (0 until numIterations).foreach { i =>
      val testHypers = hyperGrids.map { case (n, v) =>
        n -> Random.shuffle(v).head
      }
      val testLearner = base.setHypers(testHypers)
      val res = testLearner.train(trainingData).asInstanceOf[TrainingResult with hasLoss]
      /* Keep track of the best */
      if (res.getLoss() < loss) {
        best = testHypers
        loss = res.getLoss()
        println(s"Improved the loss to ${loss} with ${best}")
      }
    }
    (best, loss)
  }
}
