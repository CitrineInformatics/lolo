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
    val Y = utils.makeLinearCorrelatedData(X, rho)
    assert(math.abs(utils.correlation(X, Y) - rho) < 1e-8)
  }

}
