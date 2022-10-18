package io.citrine.lolo

case class TrainingRow[+T](inputs: Vector[Any], labels: T, weight: Double) {
  def withWeight(weight: Double): TrainingRow[T] = copy(weight = weight)
}

object TrainingRow {

  def extractLabel[T](row: TrainingRow[Vector[Any]], idx: Int): TrainingRow[T] =
    TrainingRow(row.inputs, row.labels(idx).asInstanceOf[T], row.weight)
}
