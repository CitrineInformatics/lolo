package io.citrine.lolo

trait TrainingRow[T] {
  def features: Vector[Any]
  def weight: Double
}

object TrainingRow {
  case class SingleTaskTrainingRow[T](features: Vector[Any], label: T, weight: Double) extends TrainingRow[T]
  case class MultiTaskTrainingRow(features: Vector[Any], labels: Vector[Any], weight: Double)
      extends TrainingRow[Vector[Any]]
}
