package io.citrine.lolo.trees.splits

import org.junit.Test

class SplitTest {

  /**
    * Make sure that splits on NaN will distinguish between NaN and non-NaN values.
    */
  @Test
  def testSplitNaN(): Unit = {
    val split = RealSplit(0, Double.NaN)
    assert(
      split.turnLeft(Vector(0.0)) != split.turnLeft(Vector(Double.NaN)),
      "NaN and non-Nan values are split in the same direction."
    )
  }

}
