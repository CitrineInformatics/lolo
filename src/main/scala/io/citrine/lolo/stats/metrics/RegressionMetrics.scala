package io.citrine.lolo.stats.metrics

object RegressionMetrics {

  /** Compute the root mean squared error. */
  def RMSE(predictedVsActual: Seq[(Vector[Any], Double, Double)]): Double = {
    math.sqrt(predictedVsActual.map { case (_, p, a) => math.pow(p - a, 2.0) }.sum / predictedVsActual.length)
  }

}
