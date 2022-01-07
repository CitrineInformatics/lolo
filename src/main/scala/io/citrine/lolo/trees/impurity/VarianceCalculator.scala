package io.citrine.lolo.trees.impurity

/** Calculat the weighted variance, which is \sum w_i * (x_i - \bar{x})^2, where \bar{x} is the weighted mean of x
  *
  * @param totalSum
  *   weighted sum of the labels
  * @param totalSquareSum
  *   weighted sum of the squares of the labels
  * @param totalWeight
  *   sum of the weights
  */
class VarianceCalculator(
    totalSum: Double,
    totalSquareSum: Double,
    totalWeight: Double
) extends ImpurityCalculator[Double] {

  def add(value: Double, weight: Double): Double = {
    if (!value.isNaN) {
      leftSum += value * weight
      leftWeight += weight
    }
    getImpurity
  }

  def remove(value: Double, weight: Double): Double = {
    if (!value.isNaN) {
      leftSum -= value * weight
      leftWeight -= weight
    }
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
    if (totalWeight == 0.0) {
      0.0
    } else if (rightWeight == 0.0 || leftWeight == 0.0) {
      totalSquareSum - totalSum * totalSum / totalWeight
    } else {
      totalSquareSum - leftSum * leftSum / leftWeight - rightSum * rightSum / rightWeight
    }
  }

  private var leftSum: Double = 0.0
  private var leftWeight: Double = 0.0
}

/** Companion object
  */
object VarianceCalculator {

  /** Build a variance calculator for labels and weights
    *
    * @param labels
    *   to build calculator for
    * @param weights
    *   to build calculator for
    * @return
    *   VarianceCalculator for these labels and weights
    */
  def build(labels: Seq[Double], weights: Seq[Double]): VarianceCalculator = {
    // be sure to filter out "missing" labels, which are NaN
    val config: (Double, Double, Double) = labels
      .zip(weights)
      .filterNot(_._1.isNaN())
      .map { case (l, w) =>
        (w * l, w * l * l, w)
      }
      .fold((0.0, 0.0, 0.0)) { (p1: (Double, Double, Double), p2: (Double, Double, Double)) =>
        (p1._1 + p2._1, p1._2 + p2._2, p1._3 + p2._3)
      }
    new VarianceCalculator(config._1, config._2, config._3)
  }
}
