package io.citrine.lolo.hypers

import io.citrine.lolo.Learner

/**
  * Brute force search over the grid of hypers
  *
  * Created by maxhutch on 12/7/16.
  *
  * @param base learner to optimize parameters for
  */
class GridHyperOptimizer(base: Learner) extends HyperOptimizer(base) {

  /**
    * Search by enumerating every combination of hyper values
    *
    * @param trainingData  the data to train/test on
    * @param numIterations ignored, since this is a brute force search
    * @return the best hyper map found in the search space
    */
  override def optimize(trainingData: Seq[(Vector[Any], Any)], numIterations: Int = 1): (Map[String, Any], Double) = {
    var best: Map[String, Any] = Map()
    var loss = Double.MaxValue
    /* Get the size of each dimension for index arithmetic */
    val sizes = hyperGrids.mapValues(_.size)

    /* We are flattening the loops into a single look with index arithmetic */
    (0 until sizes.values.product).foreach { i =>
      var j = i
      var testHypers: Map[String, Any] = Map()
      /* For each hyper dimension, pull an index with mod and then shift */
      sizes.foreach { case (name, size) =>
        val index = j % size
        testHypers = testHypers + ((name, hyperGrids(name)(index)))
        j = j / size
      }

      /* Set up a learner with these parameters and compute the loss */
      val testLearner = base.setHypers(testHypers)
      val res = testLearner.train(trainingData)
      if (res.getLoss().isEmpty) {
        throw new IllegalArgumentException("Trying to optimize hyper-paramters for a learner without getLoss")
      }
      val thisLoss = res.getLoss().get

      /* Save if it is an improvement */
      if (thisLoss < loss) {
        best = testHypers
        loss = thisLoss
        println(s"Improved the loss to ${loss} with ${best}")
      }
    }
    (best, loss)
  }
}
