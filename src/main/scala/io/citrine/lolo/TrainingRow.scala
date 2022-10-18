package io.citrine.lolo

case class TrainingRow[+T](inputs: Vector[Any], label: T, weight: Double) {

  /** Cast the training row as a tuple of (inputs, label, weight). */
  def asTuple: (Vector[Any], T, Double) = (inputs, label, weight)

  /** Return a new row holding up to the first `n` features from the input vector. */
  def takeInputs(n: Int): TrainingRow[T] = copy(inputs = inputs.take(n))
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
