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
  def buildMulti(values: Seq[Vector[Any]]): Seq[Option[Standardization]] = {
    values.headOption
      .map { rep =>
        rep.indices.map { i =>
          rep(i) match {
            case _: Double => Some(Standardization.build(values.map(r => r(i).asInstanceOf[Double])))
            case _: Any    => None
          }
        }
      }
      .getOrElse(Seq.empty)
  }

  /**
    * Apply the standardizations to vectors, which should result in an output with zero mean and unit variance
    *
    * @param input to standardize
    * @param trans transformtions to apply.  None means no transformation
    * @return sequence of standardized vectors
    */
  def applyMulti(input: Vector[Any], trans: Seq[Option[Standardization]]): Vector[Any] = {
    input.zip(trans).map {
      case (x: Double, Some(t)) => t.apply(x)
      case (x, _)               => x
    }
  }

  /**
    * Invert the standardizations on vectors.
    *
    * @param input to invert the standardization
    * @param trans transformations to un-apply. None means no transformation
    * @return sequence of restored vectors
    */
  def invertMulti(input: Vector[Any], trans: Seq[Option[Standardization]]): Vector[Any] = {
    input.zip(trans).map {
      case (x: Double, Some(t)) => t.invert(x)
      case (x, _)               => x
    }
  }
}
