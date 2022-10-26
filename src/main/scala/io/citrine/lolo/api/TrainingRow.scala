package io.citrine.lolo.api

/**
  * A bundle of (inputs, label, weight) used for training a [[Learner]].
  *
  * @param inputs vector of mixed-type input features
  * @param label  target variable to train a model on
  * @param weight of the training row
  * @tparam T the type of the label data
  */
case class TrainingRow[+T](inputs: Vector[Any], label: T, weight: Double = 1.0) {

  /** Cast the training row as a tuple of (inputs, label, weight). */
  def asTuple: (Vector[Any], T, Double) = (inputs, label, weight)

  /** Return a new row updated with the provided inputs. */
  def withInputs(inputs: Vector[Any]): TrainingRow[T] = copy(inputs = inputs)

  /** Return a new row updated with the provided label. */
  def withLabel[T2](label: T2): TrainingRow[T2] = copy(label = label)

  /** Return a new row updated with the provided weight. */
  def withWeight(weight: Double): TrainingRow[T] = copy(weight = weight)

  /** Return a new row with the function applied to the original inputs. */
  def mapInputs(f: Vector[Any] => Vector[Any]): TrainingRow[T] = copy(inputs = f(inputs))

  /** Return a new row with the function applied to the original label. */
  def mapLabel[T2](f: T => T2): TrainingRow[T2] = copy(label = f(label))

  /** Return a new row with the function applied to the original weight. */
  def mapWeight(f: Double => Double): TrainingRow[T] = copy(weight = f(weight))
}

object TrainingRow {

  /** Build a sequence of training from the provided data and weights. */
  def build[T](data: Seq[(Vector[Any], T)], weights: Option[Seq[Double]] = None): Seq[TrainingRow[T]] = {
    val actualWeights = weights.getOrElse(Seq.fill(data.length)(1.0))
    data.zip(actualWeights).map {
      case ((i, l), w) => TrainingRow(i, l, w)
    }
  }

  /** Build a sequence of training from the provided data and weights. */
  def build[T](data: Seq[(Vector[Any], T, Double)]): Seq[TrainingRow[T]] = {
    data.map { case (i, l, w) => TrainingRow(i, l, w) }
  }
}
