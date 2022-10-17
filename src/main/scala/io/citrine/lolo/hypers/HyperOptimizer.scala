package io.citrine.lolo.hypers

import io.citrine.lolo.Learner
import io.citrine.random.Random

/**
  * Base class for hyperparameter optimizers
  *
  * They take a range of hypers as a Map[(String, Seq[Any])] and output the best map and loss
  */
abstract class HyperOptimizer() {

  /**
    * Add a 1D hyper range to the space searched by this optimizer
    *
    * @param name   of the hyper
    * @param values it takes, enumerated as a seq
    * @return calling instance
    */
  def addHyperGrid(name: String, values: Seq[Any]): this.type = {
    hyperGrids = hyperGrids + ((name, values))
    this
  }

  /** The search space */
  var hyperGrids: Map[String, Seq[Any]] = Map()

  /**
    * Optimize the hypers over the defined search space
    *
    * @param trainingData  the data to train/test on
    * @param numIterations to take before terminating
    * @return the best hyper map found in give iterations and the corresponding loss
    */
  def optimize[T](
      trainingData: Seq[(Vector[Any], T)],
      numIterations: Int = 8,
      builder: Map[String, Any] => Learner[T],
      rng: Random = Random()
  ): (Map[String, Any], Double)
}
