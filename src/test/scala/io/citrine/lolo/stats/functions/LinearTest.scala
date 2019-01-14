package io.citrine.lolo.stats.functions

import org.junit.Test

class LinearTest {

  @Test
  def testSimpleLinear(): Unit = {
    val func: Linear = Linear(Seq(1.0))
    println(func(Seq(1.0, 2.0, 3.0)))
  }

}
