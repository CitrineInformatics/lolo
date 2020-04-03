package io.citrine.lolo.stats.functions

import scala.util.Random

/**
  * Linear function, defined by its gradient
  *
  * The gradient is implicitly zero-padded at the end
  * @param gradient of the linear function
  */
case class Linear(gradient: Seq[Double]) {
  def apply(x: Seq[Double]): Double = {
    x.zip(gradient).map{case (a, b) => a * b}.sum
  }
}

object Linear {
  /**
    * Generate a linear function in a random direction
    * @param nDim dimension of the function, which is the length of the gradient
    * @param magnitude of the gradient (default: 1)
    * @return linear function with specified gradient magnitude in a random direction
    */
  def randomDirection(nDim: Int, magnitude: Double = 1.0, rng: Random = Random): Linear = {
    val gradient = {
      // Draw guassian so the direction is random rather than preferring corners
      val unnormalized = Seq.fill(nDim){rng.nextGaussian()}
      val norm = Math.sqrt(unnormalized.map(Math.pow(_, 2)).sum)
      unnormalized.map(x => x * magnitude / norm)
    }
    Linear(gradient)
  }

  /**
    * Generate a linear function that has a gradient proportional to the one-vector, scaled to the requested magnitude
    * @param nDim dimension of the function, which is the length of the gradient
    * @param magnitude of the gradient (default: 1)
    * @return linear function with specified gradient magnitude that points in the (1, 1, ...) direction
    */
  def offDiagonal(nDim: Int, magnitude: Double = 1.0): Linear = {
    val gradient = Seq.fill(nDim)(magnitude / Math.sqrt(nDim))
    Linear(gradient)
  }
}
