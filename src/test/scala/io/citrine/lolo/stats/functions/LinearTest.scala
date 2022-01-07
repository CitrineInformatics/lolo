package io.citrine.lolo.stats.functions

import org.junit.Test

class LinearTest {

  /** Simple test of linear function
    */
  @Test
  def testSimpleLinear(): Unit = {
    val func: Linear = Linear(Seq(1.0))
    assert(func(Seq(1.0, 2.0, 3.0)) == 1.0)
  }

}
