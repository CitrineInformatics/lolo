package io.citrine.lolo.trees.splits

class GiniCalculator() {

  def add(value: Double, weight: Double): Double = {
    leftSum += value * weight
    leftWeight += weight
    getImpurity
  }

  def remove(value: Double, weight: Double): Double = {
    leftSum -= value * weight
    leftWeight -= weight
    getImpurity
  }

  def reset(): Unit = {
    leftSum = 0.0
    leftWeight = 0.0
  }

  def copy(): GiniCalculator = {
    new GiniCalculator(totalSum, totalSquareSum, totalWeight)
  }

  def getImpurity: Double = {
    val rightSum = totalSum - leftSum
    val rightWeight = totalWeight - leftWeight
    totalSquareSum - leftSum * leftSum / leftWeight - rightSum * rightSum / rightWeight
  }

  private var leftSum: Double = 0.0
  private var leftWeight: Double = 0.0
}


