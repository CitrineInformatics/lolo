package io.citrine.lolo.hypers

import io.citrine.lolo.{Learner, TrainingRow}
import io.citrine.random.Random

/**
  * Search for hypers by randomly sampling the search space
  *
  * This optimizer can be evaluated multiple times and will persist the best results across those
  * calls.
  */
class RandomHyperOptimizer() extends HyperOptimizer {

  /** Keep track of the best hypers outside of the optimize call so it persists across calls */
  var best: Map[String, Any] = Map()

  /** Likewise with the lowest loss */
  var loss: Double = Double.MaxValue

  /**
    * Search over numIterations random draws for the search space
    *
    * @param trainingData  the data to train/test on
    * @param numIterations number of draws to take
    * @return the best hyper map found in give iterations and the corresponding loss
    */
  override def optimize[T](
      trainingData: Seq[TrainingRow[T]],
      numIterations: Int,
      builder: Map[String, Any] => Learner[T],
      rng: Random
  ): (Map[String, Any], Double) = {
    /* Just draw numIteration times */
    (0 until numIterations).foreach { _ =>
      val testHypers = hyperGrids.map {
        case (n, v) =>
          n -> rng.shuffle(v).head
      }
      val testLearner = builder(testHypers)
      val res = testLearner.train(trainingData, rng = rng)
      if (res.getLoss().isEmpty) {
        throw new IllegalArgumentException("Trying to optimize hyper-paramters for a learner without getLoss")
      }
      val thisLoss = res.getLoss().get
      /* Keep track of the best */
      if (thisLoss < loss) {
        best = testHypers
        loss = thisLoss
        println(s"Improved the loss to $loss with $best")
      }
    }
    (best, loss)
  }
}
