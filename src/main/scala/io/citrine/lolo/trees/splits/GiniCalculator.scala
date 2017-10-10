package io.citrine.lolo.trees.splits

import scala.collection.mutable

class GiniCalculator(
                      totalCategoryWeights: Map[Char, Double],
                      totalSquareSum: Double,
                      totalWeight: Double
                    ) extends ImpurityCalculator[Char] {

  def add(value: Char, weight: Double): Double = {
    val wl = leftCategoryWeights.getOrElse(value, 0.0)
    leftSquareSum += weight * (weight + 2 * wl)
    val wr = totalCategoryWeights(value) - wl
    rightSquareSum += weight * (weight - 2 * wr)
    leftCategoryWeights(value) = wl + weight
    leftWeight += weight

    getImpurity
  }

  def remove(value: Char, weight: Double): Double = {
    val wl = leftCategoryWeights.getOrElse(value, 0.0)
    leftSquareSum += weight * (weight - 2 * wl)
    val wr = totalCategoryWeights(value) - wl
    rightSquareSum += weight * (weight + 2 * wr)
    leftCategoryWeights(value) = wl - weight
    leftWeight -= weight

    getImpurity
  }

  def reset(): Unit = {
    leftCategoryWeights.clear()
    leftWeight = 0.0
    leftSquareSum = 0.0
    rightSquareSum = totalSquareSum
  }

  def copy(): GiniCalculator = {
    new GiniCalculator(totalCategoryWeights, totalSquareSum, totalWeight)
  }

  def getImpurity: Double = {
    if (leftWeight == 0 || leftWeight == totalWeight) {
      totalWeight - totalSquareSum / totalWeight
    } else {
      totalWeight - leftSquareSum / leftWeight - rightSquareSum / (totalWeight - leftWeight)
    }
  }

  private val leftCategoryWeights = mutable.Map[Char, Double]()
  private var leftWeight: Double = 0.0
  private var leftSquareSum: Double = 0.0
  private var rightSquareSum: Double = totalSquareSum
}

object GiniCalculator {
  def build(data: Seq[(Char, Double)]): GiniCalculator = {
    val totalCategoryWeights = data.groupBy(_._1).mapValues(_.map(_._2).sum)
    val totalSquareSum = totalCategoryWeights.values.map(v => Math.pow(v, 2)).sum
    val totalWeight = totalCategoryWeights.values.sum
    new GiniCalculator(totalCategoryWeights, totalSquareSum, totalWeight)
  }
}


