package io.citrine.lolo

case class TrainingRow[+T](inputs: Vector[Any], label: T, weight: Double) {

  def asTuple: (Vector[Any], T, Double) = (inputs, label, weight)

  def withInputs(inputs: Vector[Any]): TrainingRow[T] = copy(inputs = inputs)

  def withWeight(weight: Double): TrainingRow[T] = copy(weight = weight)
}

object TrainingRow {

  /**
    * Extract the `idx` label from a multi-task training row and cast it to type `T`.
    *
    * @param row containing multi-task label vectors
    * @param idx of the label to extract
    * @tparam T of the extracted label
    * @return a TrainingRow[T] for the extracted label type
    */
  def extractLabel[T](row: TrainingRow[Vector[Any]], idx: Int): TrainingRow[T] =
    TrainingRow(row.inputs, row.label(idx).asInstanceOf[T], row.weight)
}
