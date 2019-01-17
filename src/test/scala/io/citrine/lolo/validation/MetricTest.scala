package io.citrine.lolo.validation

import org.junit.Test

import scala.util.Random

class MetricTest {

  @Test
  def testUncertaintyCorrelationBaseline(): Unit = {
    val N = 2 << 16
    val sigmas = Seq.fill(N){
      Math.abs(Random.nextGaussian() + 100)
    }
    val pua = sigmas.map{sigma =>
      val actual = Random.nextDouble()
      val error = Random.nextGaussian() * sigma
      (actual + error, sigma, actual)
    }
    println(UncertaintyCorrelation.computeFromPredictedUncertaintyActual(pua))
  }

}
