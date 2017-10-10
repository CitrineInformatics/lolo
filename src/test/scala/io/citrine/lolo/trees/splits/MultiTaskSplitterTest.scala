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

  /**
    * Make sure the split functions when one of the labels is completely missing
    */
  @Test
  def testAllEmptyLabels(): Unit = {
    val inputs = Seq(
      Vector(0.4566490222691504, 0.21730793346810628, 0.08289198941227605),
      Vector(0.0025661310641983404, 0.21416453420489545, 0.15783628254609094)
    )
    val labels = Vector(Array(1.238, null), Array(1.180, null))
    val weights = Vector(1.0, 1.0)
    val data = inputs.indices.map{i =>
      (inputs(i), labels(i).asInstanceOf[Array[AnyVal]], weights(i))
    }
    val (pivot, impurity) = MultiTaskSplitter.getBestSplit(data, data.head._1.size, 1)
    assert(!pivot.isInstanceOf[NoSplit])
  }
}
