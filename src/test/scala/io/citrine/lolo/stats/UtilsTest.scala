package io.citrine.lolo.stats

import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

class UtilsTest {

  /** Test that a linearly correlated data set has the expected correlation coefficient. */
  @Test
  def testLinearCorrelatedData(): Unit = {
    val rng = new Random()
    val rho = rng.nextDouble() * 2.0 - 1.0
    val X = Seq.fill(24)(rng.nextDouble())
    val Y = StatsUtils.makeLinearCorrelatedData(X, rho)
    assert(math.abs(StatsUtils.correlation(X, Y) - rho) < 1e-8)
  }

  /** Test the implementation of median */
  @Test
  def testMedian(): Unit = {
    assert(StatsUtils.median(Seq(14.0, 17.0, 12.0, 13.0, 13.0)) == 13.0)
    assert(StatsUtils.median(Seq(1.0, 0.0, 3.0, 4.0)) == 2.0)
  }

}
