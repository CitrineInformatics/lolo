package io.citrine.lolo.trees.splits

import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 12/1/16.
  */
@Test
class MultiTaskSplitterTest {

  /**
    * Test calculation of impurity on a known set
    */
  @Test
  def testImpurity(): Unit = {
    val data = Seq(
      (Array[AnyVal](0.0, 1.toChar), 1.0),
      (Array[AnyVal](1.0, 2.toChar), 1.0),
      (Array[AnyVal](100.0, 3.toChar), 0.0)
    )

    val impurity = MultiTaskSplitter.computeImpurity(data)
    assert(impurity == (0.25 + 0.5) * 2.0)
  }

  /**
    * Test that the real split goes in the right place
    */
  @Test
  def testBestRealSplit(): Unit = {
    val data = Seq(
      (Vector(0.0), Array[AnyVal](0.0, 1.toChar), 1.0),
      (Vector(1.0), Array[AnyVal](1.0, 2.toChar), 1.0),
      (Vector(2.0), Array[AnyVal](2.0, 2.toChar), 1.0),
      (Vector(100.0), Array[AnyVal](200.0, 3.toChar), 0.0)
    )
    val calculator = MultiImpurityCalculator.build(data.map(_._2), data.map(_._3))

    val (pivot, impurity) = MultiTaskSplitter.getBestRealSplit(data, calculator, 0, 1)
    val turns = data.map(x => pivot.turnLeft(x._1))
    assert(turns == Seq(true, false, false, false))
    assert(impurity == 0.5)
  }

  /**
    * Test that a categorical split goes in the right place
    */
  @Test
  def testBestCategoricalSplit(): Unit = {
    val data = Seq(
      (Vector(1.toChar), Array[AnyVal](0.0, 1.toChar), 1.0),
      (Vector(2.toChar), Array[AnyVal](1.0, 2.toChar), 1.0),
      (Vector(2.toChar), Array[AnyVal](2.0, 2.toChar), 1.0),
      (Vector(3.toChar), Array[AnyVal](200.0, 3.toChar), 0.0)
    )
    val calculator = MultiImpurityCalculator.build(data.map(_._2), data.map(_._3))

    val (pivot, impurity) = MultiTaskSplitter.getBestCategoricalSplit(data, calculator, 0, 1)
    val turns = data.map(x => pivot.turnLeft(x._1)).take(3)
    println(turns)
    assert(turns == Seq(true, false, false))
    assert(impurity == 0.5)
  }
}
