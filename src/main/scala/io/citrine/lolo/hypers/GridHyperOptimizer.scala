package io.citrine.lolo.hypers

import io.citrine.lolo.api.{Learner, TrainingRow}
import io.citrine.random.Random

/** Brute force search over the grid of hypers. */
case class GridHyperOptimizer() extends HyperOptimizer {

  /**
    * Search by enumerating every combination of hyper values
    *
    * @param trainingData  the data to train/test on
    * @param numIterations ignored, since this is a brute force search
    * @return the best hyper map found in the search space
    */
  override def optimize[T](
      trainingData: Seq[TrainingRow[T]],
      numIterations: Int = 1,
      builder: Map[String, Any] => Learner[T],
      rng: Random
  ): (Map[String, Any], Double) = {
    var best: Map[String, Any] = Map()
    var loss = Double.MaxValue
    /* Get the size of each dimension for index arithmetic */
    val sizes = hyperGrids.view.mapValues(_.size)

    /* We are flattening the loops into a single look with index arithmetic */
    (0 until sizes.values.product).foreach { i =>
      var j = i
      var testHypers: Map[String, Any] = Map()
      /* For each hyper dimension, pull an index with mod and then shift */
      sizes.foreach {
        case (name, size) =>
          val index = j % size
          testHypers = testHypers + ((name, hyperGrids(name)(index)))
          j = j / size
      }

      // Set up a learner with these parameters and compute the loss
      val testLearner = builder(testHypers)
      val res = testLearner.train(trainingData, rng = rng)
      if (res.loss.isEmpty) {
        throw new IllegalArgumentException("Trying to optimize hyper-parameters for a learner without getLoss")
      }
      val thisLoss = res.loss.get

      /* Save if it is an improvement */
      if (thisLoss < loss) {
        best = testHypers
        loss = thisLoss
        println(s"Improved the loss to $loss with $best")
      }
    }
    (best, loss)
  }
}
