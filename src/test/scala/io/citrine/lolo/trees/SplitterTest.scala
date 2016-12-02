package io.citrine.lolo.trees

import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 12/1/16.
  */
@Test
class SplitterTest {

  /**
    * Test different variance formulations
    */
  @Test
  def testTotalVarianceCalculation(): Unit = {
    val values = (0 until 16).map(i => Random.nextDouble())
    val (left, right) = values.partition(f => Random.nextBoolean())

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
      (totalNum - leftNum) * ((totalSquareSum - leftSquareSum) / (totalNum - leftNum) - Math.pow((totalSum - leftSum) / (totalNum - leftNum), 2))) / totalNum

    val totalVarianceSuperShort = (
      -Math.pow(leftSum, 2) / leftNum - Math.pow((totalSum - leftSum), 2) / (totalNum - leftNum)
      ) / totalNum + (totalSquareSum / totalNum)

    assert(Math.abs(totalVariance - totalVarianceShorter) < 1.0e-9, s"${totalVariance} != ${totalVarianceShorter}")
    assert(Math.abs(totalVariance - totalVarianceSuperShort) < 1.0e-9, s"${totalVariance} != ${totalVarianceSuperShort}")
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
    new SplitterTest().testTotalVarianceCalculation()
  }
}
