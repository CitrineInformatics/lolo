package io.citrine.lolo.transformers.standardization

/** A package of the shift/scale required to standardize values to zero mean and unit variance. */
case class Standardization(shift: Double, scale: Double) {
  require(scale > 0 && scale < Double.PositiveInfinity)

  def apply(x: Double): Double = (x - shift) / scale

  def invert(x: Double): Double = x * scale + shift
}

object Standardization {

  /**
    * The standardizations are just shifts and rescale.
    * The shift is by the mean and the re-scale is by the variance
    *
    * @param values to get a standardizer for
    * @return (shift, rescaling)
    */
  def build(values: Seq[Double]): Standardization = {
    val mean = values.sum / values.size
    val scale = math.sqrt(values.map(v => math.pow(v - mean, 2)).sum / values.size)

    // If there is zero variance, then the scaling doesn't matter; default to 1.0
    if (scale > 0 && scale < Double.PositiveInfinity) {
      Standardization(mean, scale)
    } else {
      Standardization(mean, 1.0)
    }
  }

  /**
    * Get standardization for multiple values in a vector.
    *
    * This has a different name because the JVM erases the inner type of Seq[T].
    *
    * @param values sequence of vectors to be standardized
    * @return sequence of standardization, each as an option
    */
  def buildMulti(values: Seq[Vector[Any]]): Map[Int, Standardization] = {
    val rep = values.head
    val valuesTransposed = values.transpose
    rep.zipWithIndex.collect {
      case (_: Double, idx) =>
        val doubleValues = valuesTransposed(idx).collect { case x: Double if !x.isNaN => x }
        idx -> Standardization.build(doubleValues)
    }.toMap
  }

  /**
    * Apply the standardizations to a vector, which should result in an output with zero mean and unit variance
    *
    * @param input to standardize
    * @param trans transformations to apply
    * @return a standardized vector
    */
  def applyMulti(input: Vector[Any], trans: Map[Int, Standardization]): Vector[Any] = {
    input.zipWithIndex.map {
      case (x: Double, idx) => trans.get(idx).map(_.apply(x)).getOrElse(x)
      case (x, _)           => x
    }
  }

  /**
    * Apply the standardizations to a sequence of optional values.
    *
    * @param input to apply the standardization, if the value is defined
    * @param trans transformations to un-apply
    * @return a standardized vector of optional values
    */
  def applyMultiOption(input: Vector[Option[Any]], trans: Map[Int, Standardization]): Vector[Option[Any]] = {
    input.zipWithIndex.map {
      case (Some(x: Double), idx) => Some(trans.get(idx).map(_.apply(x)).getOrElse(x))
      case (x, _)                 => x
    }
  }

  /**
    * Invert the standardization on a vector.
    *
    * @param input to invert the standardization
    * @param trans transformations to un-apply
    * @return a restored vector
    */
  def invertMulti(input: Vector[Any], trans: Map[Int, Standardization]): Vector[Any] = {
    input.zipWithIndex.map {
      case (x: Double, idx) => trans.get(idx).map(_.invert(x)).getOrElse(x)
      case (x, _)           => x
    }
  }

  /**
    * Invert the standardizations on a sequence of optional values.
    *
    * @param input to invert the standardization, if the value is defined
    * @param trans transformations to un-apply
    * @return a restored vector of optional values
    */
  def invertMultiOption(input: Vector[Option[Any]], trans: Map[Int, Standardization]): Vector[Option[Any]] = {
    input.zipWithIndex.map {
      case (Some(x: Double), idx) => Some(trans.get(idx).map(_.invert(x)).getOrElse(x))
      case (x, _)                 => x
    }
  }
}
