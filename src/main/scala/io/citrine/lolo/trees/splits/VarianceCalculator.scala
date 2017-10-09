package io.citrine.lolo.trees.splits

class VarianceCalculator(totalSum: Double, totalSquareSum: Double, totalWeight: Double) {

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

  def copy(): VarianceCalculator = {
    new VarianceCalculator(totalSum, totalSquareSum, totalWeight)
  }

  def getImpurity: Double = {
    val rightSum = totalSum - leftSum
    val rightWeight = totalWeight - leftWeight
    totalSquareSum - leftSum * leftSum / leftWeight - rightSum * rightSum / rightWeight
  }

  private var leftSum: Double = 0.0
  private var leftWeight: Double = 0.0
}

object VarianceCalculator {
  def build(labels: Seq[Double], weights: Seq[Double]): VarianceCalculator = {
    val config: (Double, Double, Double) = labels.zip(weights).map { case (l, w) =>
      (w * l, w * l * l, w)
    }.reduce { (p1: (Double, Double, Double), p2: (Double, Double, Double)) =>
      (p1._1 + p2._1, p1._2 + p2._2, p1._3 + p2._3)
    }
    new VarianceCalculator(config._1, config._2, config._3)
  }
}
