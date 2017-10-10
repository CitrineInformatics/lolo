package io.citrine.lolo.trees.splits

/**
  * Interface for an impurity calculator, which computes an impurity metric to drive a splitter
  * @tparam T
  */
trait ImpurityCalculator[T] {

  /**
    * Add a value to the left partition
    * @param value to add
    * @param weight of the value
    * @return the impurity after adding
    */
  def add(value: T, weight: Double): Double

  /**
    * Remove a value from the left partition
    * @param value to remove
    * @param weight of the value
    * @return the impurity after removing
    */
  def remove(value: T, weight: Double): Double

  /**
    * Reset the calculator such that every item is on the right
    */
  def reset(): Unit

  /**
    * Get the impurity at the current state of the calculator
    * @return impurity
    */
  def getImpurity: Double
}
