package io.citrine.lolo.trees.splits

import io.citrine.lolo.SeedRandomMixIn
import io.citrine.lolo.api.TrainingRow
import io.citrine.random.Random
import org.junit.Test

class BoltzmannSplitterTest extends SeedRandomMixIn {

  /**
    * Test that uniform labels result in "NoSplit" with zero reduced impurity
    */
  @Test
  def testZeroVariance(): Unit = {
    val splitter = BoltzmannSplitter(1.0e-9)
    val testData = Seq.fill(64) {
      val x = rng.nextDouble()
      val y = 1.0
      val weight = 1.0
      TrainingRow(Vector(x), y, weight)
    }

    val (bestSplit, improvement) = splitter.getBestSplit(testData, 1, 1)
    assert(bestSplit.isInstanceOf[NoSplit], "Can't split on zero variance")
    assert(improvement == 0, "Can't improve on no variance")
  }

  /**
    * Test that very small random signals don't lead to exceptions.
    *
    * Note that this test will return NoSplit occasionally, when the starting impurity is numerically zero.  That's OK
    * though.
    */
  @Test
  def testLowVariance(): Unit = {
    val splitter = BoltzmannSplitter(1.0e-18)
    val testData = Seq.fill(256) {
      val x = rng.nextDouble()
      val y = rng.nextGaussian() * 1.0e-9 + 1.0
      val weight = 1.0
      TrainingRow(Vector(x), y, weight)
    }

    splitter.getBestSplit(testData, 1, 1, rng = rng)
  }

}
