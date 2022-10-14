package io.citrine.lolo.bags

import io.citrine.lolo.Learner

trait Bagger[T] extends Learner[T] {}

object Bagger {

  /**
    * The minimum number of training rows in order to train a Bagger
    */
  val minimumTrainingSize: Int = 8

  /** The minimum number of training examples with a value for any given output */
  val minimumOutputCount: Int = 2

  /**
    * The minimum number of non-zero weighted training points that is based into the learner
    *
    * This requirement biases the poisson draws, but hopefully not too too much
    */
  val minimumNonzeroWeightSize: Int = 4

  protected[bags] def combineImportance(
      v1: Option[Vector[Double]],
      v2: Option[Vector[Double]]
  ): Option[Vector[Double]] = {
    (v1, v2) match {
      case (None, None)                                         => None
      case (Some(v1: Vector[Double]), Some(v2: Vector[Double])) => Some(v1.zip(v2).map(p => p._1 + p._2))
      case _                                                    => None
    }
  }
}
