package io.citrine.lolo.trees.splits

import org.junit.Test

import scala.util.Random

class BoltzmannSplitterTest {

  /**
    * Test that uniform labels result in "NoSplit" with zero reduced impurity
    */
  @Test
  def testZeroVariance(): Unit = {
    val splitter = BoltzmannSplitter(1.0e-9)
    val testData = Seq.fill(64){
      val x = Random.nextDouble()
      val y = 1.0
      val weight = 1.0
      (Vector(x), y, weight)
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
    val testData = Seq.fill(256){
      val x = Random.nextDouble()
      val y = Random.nextGaussian() * 1.0e-9 + 1.0
      val weight = 1.0
      (Vector(x), y, weight)
    }

    splitter.getBestSplit(testData, 1, 1)
  }

}
