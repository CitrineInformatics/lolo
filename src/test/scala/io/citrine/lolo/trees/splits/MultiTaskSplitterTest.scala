package io.citrine.lolo.trees.splits

import io.citrine.lolo.SeedRandomMixIn
import io.citrine.lolo.trees.impurity.MultiImpurityCalculator
import org.junit.Test

/**
  * Created by maxhutch on 12/1/16.
  */
@Test
class MultiTaskSplitterTest extends SeedRandomMixIn {

  /**
    * Test that the real split goes in the right place
    */
  @Test
  def testBestRealSplit(): Unit = {
    val data = Seq(
      (Vector(0.0), Vector[AnyVal](0.0, 1.toChar), 1.0),
      (Vector(1.0), Vector[AnyVal](1.0, 2.toChar), 1.0),
      (Vector(2.0), Vector[AnyVal](2.0, 2.toChar), 1.0),
      (Vector(100.0), Vector[AnyVal](200.0, 3.toChar), 0.0)
    )
    val calculator = MultiImpurityCalculator.build(data.map(_._2), data.map(_._3))

    val splitter = MultiTaskSplitter()
    val (pivot, impurity) = Splitter.getBestRealSplit(data, calculator, 0, 1, rng = rng)
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
      (Vector(1.toChar), Vector[AnyVal](0.0, 1.toChar), 1.0),
      (Vector(2.toChar), Vector[AnyVal](1.0, 2.toChar), 1.0),
      (Vector(2.toChar), Vector[AnyVal](2.0, 2.toChar), 1.0),
      (Vector(3.toChar), Vector[AnyVal](200.0, 3.toChar), 0.0)
    )
    val calculator = MultiImpurityCalculator.build(data.map(_._2), data.map(_._3))

    val splitter = MultiTaskSplitter()
    val (pivot, impurity) = splitter.getBestCategoricalSplit(data, calculator, 0, 1)
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
    val labels = Vector(Vector(1.238, null), Vector(1.180, null))
    val weights = Vector(1.0, 1.0)
    val data = inputs.indices.map { i =>
      (inputs(i), labels(i).asInstanceOf[Vector[AnyVal]], weights(i))
    }
    val splitter = MultiTaskSplitter()
    val (pivot, _) = splitter.getBestSplit(data, data.head._1.size, 1, rng = rng)
    assert(!pivot.isInstanceOf[NoSplit])
  }
}
