package io.citrine.lolo.trees.splits

trait ImpurityCalculator[T] {

  def add(value: T, weight: Double): Double

  def remove(value: T, weight: Double): Double

  def reset(): Unit

  def getImpurity: Double
}




