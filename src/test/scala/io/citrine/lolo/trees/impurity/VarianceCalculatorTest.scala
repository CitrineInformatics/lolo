package io.citrine.lolo.trees.impurity

import org.junit.Test

/** Created by maxhutch on 12/1/16.
  */
@Test
class VarianceCalculatorTest {

  /** Test that a calculator with no data has a weighted impurity of zero
    */
  @Test
  def testEmpty(): Unit = {
    val calculator = VarianceCalculator.build(Seq(), Seq())
    assert(calculator.getImpurity == 0.0)
  }
}
