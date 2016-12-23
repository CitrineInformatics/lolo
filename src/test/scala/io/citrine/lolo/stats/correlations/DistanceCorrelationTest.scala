package io.citrine.lolo.stats.correlations

import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 12/23/16.
  */
@Test
class DistanceCorrelationTest {

  val N: Int = 256

  def dist(x: Double, y: Double): Double = Math.abs(x - y)

  @Test
  def testIndependent(): Unit = {
    val X: Seq[Double] = Seq.tabulate(N)(i => Random.nextGaussian())
    val Y: Seq[Double] = Seq.tabulate(N)(i => Random.nextGaussian())
    val dcorr = DistanceCorrelation.distanceCorrelation(X, Y, dist)

    assert(dcorr < 0.15, s"dCorr for indepnedent vars is ${dcorr}" )
  }


  @Test
  def testLinear(): Unit = {
    val X: Seq[Double] = Seq.tabulate(N)(i => Random.nextDouble())
    val Y: Seq[Double] = X.map(v => v * 10)
    val dcorr = DistanceCorrelation.distanceCorrelation(X, Y, dist)

    assert(1.0 - dcorr < 1.0e-5, s"dCorr for linear is ${dcorr}")
  }

  @Test
  def testQuadratic(): Unit = {
    val X: Seq[Double] = Seq.tabulate(N)(i => Random.nextDouble() - .5)
    val Y: Seq[Double] = X.map(v => v * v * 2)
    val dcorr = DistanceCorrelation.distanceCorrelation(X, Y, dist)

    assert(dcorr > 0.4, s"dCorr for quad is ${dcorr}")
  }

  @Test
  def testSin(): Unit = {
    val X: Seq[Double] = Seq.tabulate(N)(i => Random.nextDouble() - .5)
    val Y: Seq[Double] = X.map(v => Math.sin(2 * Math.PI * v))
    val dcorr = DistanceCorrelation.distanceCorrelation(X, Y, dist)

    assert(dcorr > 0.8, s"dCorr for sin is ${dcorr}")
  }
}

object DistanceCorrelationTest {
  def main(argv: Array[String]): Unit = {
    new DistanceCorrelationTest().testIndependent()
    new DistanceCorrelationTest().testLinear()
    new DistanceCorrelationTest().testQuadratic()
    new DistanceCorrelationTest().testSin()
  }
}
