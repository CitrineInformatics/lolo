package io.citrine.lolo.hypers

import io.citrine.lolo.Learner

/** Base class for hyperparameter optimizers
  *
  * They take a range of hypers as a Map[(String, Seq[Any])] and output the best map and loss Created by maxhutch on
  * 12/8/16.
  */
abstract class HyperOptimizer() {

  /** Add a 1D hyper range to the space searched by this optimizer
    *
    * @param name
    *   of the hyper
    * @param values
    *   it takes, enumerated as a seq
    * @return
    *   calling instance
    */
  def addHyperGrid(name: String, values: Seq[Any]): this.type = {
    hyperGrids = hyperGrids + ((name, values))
    this
  }

  /** The search space */
  var hyperGrids: Map[String, Seq[Any]] = Map()

  /** Optimize the hypers over the defined search space
    *
    * @param trainingData
    *   the data to train/test on
    * @param numIterations
    *   to take before terminating
    * @return
    *   the best hyper map found in give iterations and the corresponding loss
    */
  def optimize(
      trainingData: Seq[(Vector[Any], Any)],
      numIterations: Int = 8,
      builder: Map[String, Any] => Learner
  ): (Map[String, Any], Double)
}
