package io.citrine.lolo.stats.functions

import scala.util.Random

case class Linear(gradient: Seq[Double]) {
  def apply(x: Seq[Double]): Double = {
    x.zip(gradient).map{case (a, b) => a * b}.sum
  }
}

object Linear {
  def randomDirection(nDim: Int, magnitude: Double = 1.0): Linear = {
    val gradient = {
      val unnormalized = Seq.fill(nDim){Random.nextGaussian()}
      val norm = Math.sqrt(unnormalized.map(Math.pow(_, 2)).sum)
      unnormalized.map(x => x * magnitude / norm)
    }
    Linear(gradient)
  }

  def offDiagonal(nDim: Int, magnitude: Double = 1.0): Linear = {
    val gradient = Seq.fill(nDim)(magnitude / Math.sqrt(nDim))
    Linear(gradient)
  }
}
