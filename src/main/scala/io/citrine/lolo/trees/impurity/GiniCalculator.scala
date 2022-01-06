package io.citrine.lolo.trees.impurity

/**
  * Calculate the weighted Gini Impurity: weight * (1 - \sum_j f_j^2), where f_j is the frequency of the jth label
  *
  * @param totalCategoryWeights the total weight of each label
  * @param totalSquareSum       the sum of the squares of the weights
  * @param totalWeight          the total weight over all the labels
  */
class GiniCalculator(
    totalCategoryWeights: Array[Double],
    totalSquareSum: Double,
    totalWeight: Double
) extends ImpurityCalculator[Char] {

  def add(value: Char, weight: Double): Double = {
    if (value > 0) {
      val wl = leftCategoryWeights(value)
      leftCategoryWeights(value) = wl + weight

      leftSquareSum += weight * (weight + 2 * wl)
      val wr = totalCategoryWeights(value) - wl
      rightSquareSum += weight * (weight - 2 * wr)
      leftWeight += weight
    }

    getImpurity
  }

  def remove(value: Char, weight: Double): Double = {
    if (value > 0) {
      val wl = leftCategoryWeights(value)
      leftCategoryWeights(value) = wl - weight

      leftSquareSum += weight * (weight - 2 * wl)
      val wr = totalCategoryWeights(value) - wl
      rightSquareSum += weight * (weight + 2 * wr)
      leftWeight -= weight
    }

    getImpurity
  }

  def reset(): Unit = {
    leftCategoryWeights.indices.foreach { i => leftCategoryWeights(i) = 0.0 }
    leftWeight = 0.0
    leftSquareSum = 0.0
    rightSquareSum = totalSquareSum
  }

  def copy(): GiniCalculator = {
    new GiniCalculator(totalCategoryWeights, totalSquareSum, totalWeight)
  }

  def getImpurity: Double = {
    if (totalWeight == 0) {
      0.0
    } else if (leftSquareSum == 0 || rightSquareSum == 0) {
      totalWeight - totalSquareSum / totalWeight
    } else {
      totalWeight - leftSquareSum / leftWeight - rightSquareSum / (totalWeight - leftWeight)
    }
  }

  private val leftCategoryWeights = new Array[Double](totalCategoryWeights.size)
  private var leftWeight: Double = 0.0
  private var leftSquareSum: Double = 0.0
  private var rightSquareSum: Double = totalSquareSum
}

/**
  * Companion object
  */
object GiniCalculator {

  /**
    * Build a GiniCalculator from weighted data
    *
    * @param data to build the calculator for
    * @return a GiniCalculator
    */
  def build(data: Seq[(Char, Double)]): GiniCalculator = {
    // Be sure to filter out missing labels, which are marked as 0.toChar
    val totalCategoryWeights = data.filter(_._1 > 0).groupBy(_._1).mapValues(_.map(_._2).sum)
    if (totalCategoryWeights.isEmpty) {
      return new GiniCalculator(Array.empty[Double], 0.0, 0.0)
    }

    val weightsArray = new Array[Double](totalCategoryWeights.keySet.max + 1)
    val totalSquareSum = totalCategoryWeights.map {
      case (k, v) =>
        weightsArray(k) = v
        Math.pow(v, 2)
    }.sum
    val totalWeight = totalCategoryWeights.values.sum
    new GiniCalculator(weightsArray, totalSquareSum, totalWeight)
  }
}
