package io.citrine.lolo.trees.impurity

import org.junit.Test

@Test
class MultiImpurityCalculatorTest {

  /**
    * Test calculation of impurity on a known set
    */
  @Test
  def testImpurity(): Unit = {
    val data = Seq(
      (Vector[AnyVal](0.0, 1.toChar), 1.0),
      (Vector[AnyVal](1.0, 2.toChar), 1.0),
      (Vector[AnyVal](100.0, 3.toChar), 0.0)
    )
    val calculator = MultiImpurityCalculator.build(data.map(_._1), data.map(_._2))

    val impurity = calculator.getImpurity
    assert(impurity == (0.25 + 0.5) * 2.0)
  }
}
