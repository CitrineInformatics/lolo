package io.citrine.lolo.trees.splits

import io.citrine.lolo.{SeedRandomMixIn, TrainingRow}
import io.citrine.lolo.trees.impurity.VarianceCalculator
import org.junit.Test

@Test
class SplitterTest extends SeedRandomMixIn {

  /**
    * Test different variance formulations
    */
  @Test
  def testTotalVarianceCalculation(): Unit = {
    val values = (0 until 16).map(i => rng.nextDouble())
    val (left, right) = values.partition(f => rng.nextBoolean())

    val totalSquareSum = values.map(v => v * v).sum
    val totalNum = values.size
    val totalSum = values.sum

    val leftSquareSum = left.map(v => v * v).sum
    val leftNum = left.size
    val leftSum = left.sum

    val rightSquareSum = right.map(v => v * v).sum
    val rightNum = right.size
    val rightSum = right.sum

    val totalVariance = (leftNum * (leftSquareSum / leftNum - Math.pow(leftSum / leftNum, 2)) +
      rightNum * (rightSquareSum / rightNum - Math.pow(rightSum / rightNum, 2))) / totalNum
    val totalVarianceShorter = (leftNum * (leftSquareSum / leftNum - Math.pow(leftSum / leftNum, 2)) +
      (totalNum - leftNum) * ((totalSquareSum - leftSquareSum) / (totalNum - leftNum) - Math.pow(
        (totalSum - leftSum) / (totalNum - leftNum),
        2
      ))) / totalNum

    val totalVarianceSuperShort = (
      -Math.pow(leftSum, 2) / leftNum - Math.pow((totalSum - leftSum), 2) / (totalNum - leftNum)
    ) / totalNum + (totalSquareSum / totalNum)

    assert(Math.abs(totalVariance - totalVarianceShorter) < 1.0e-9, s"${totalVariance} != ${totalVarianceShorter}")
    assert(
      Math.abs(totalVariance - totalVarianceSuperShort) < 1.0e-9,
      s"${totalVariance} != ${totalVarianceSuperShort}"
    )
  }

  /**
    * Test that large features that are almost the same don't end up being split
    *
    * There are numerical issues with the features are distinct but there is no double precision
    * value in between them to split on.  This results in post-split partitions with zero size
    */
  @Test
  def testLargeDuplicates(): Unit = {
    val base: Double = 3.0e9
    val trainingData = Seq.fill(8) {
      TrainingRow(Vector(base + rng.nextDouble()), rng.nextDouble())
    }

    val calculator = new VarianceCalculator(0.0, 0.0, 8.0)

    val (_, variance) = Splitter.getBestRealSplit(trainingData, calculator, 0, 1)
    assert(variance == Double.MaxValue, "didn't expect to find a valid split")
  }

}

/** Companion driver */
object SplitterTest {

  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    new SplitterTest().testLargeDuplicates()
  }
}
