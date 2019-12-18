package io.citrine.lolo.stats.functions

object Cute {
  def fhi(x: Seq[Double]): Double = {
    Math.sin(x(0) * 2 * Math.PI) * Math.exp(- x(1) * x(1) / 0.25)
  }
}
