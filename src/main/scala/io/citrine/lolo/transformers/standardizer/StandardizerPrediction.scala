package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.PredictionResult

trait StandardizerPrediction[+T] extends PredictionResult[T] {

  /** The base prediction from the model trained on standardized data. */
  def basePrediction: PredictionResult[T]
}

case class RegressionStandardizerPrediction(
    basePrediction: PredictionResult[Double],
    outputTrans: Standardization,
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerPrediction[Double] {

  override def getExpected(): Seq[Double] = basePrediction.getExpected().map(outputTrans.invert)

  // TODO: A PredictionResult[Double] should always return a Option[Seq[Double]] for uncertainty
  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
    basePrediction.getUncertainty(includeNoise).map { x =>
      x.map(_.asInstanceOf[Double] * outputTrans.scale)
    }
  }

  override def getGradient(): Option[Seq[Vector[Double]]] = {
    basePrediction.getGradient().map { gradients =>
      gradients.map { g =>
        g.zip(inputTrans).map {
          case (y, inputStandardization) =>
            // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
            // Otherwise, just rescale by the output transformer
            val inputScale = inputStandardization.map(_.scale).getOrElse(1.0)
            val outputScale = outputTrans.scale
            y * outputScale / inputScale
        }
      }
    }
  }
}

case class ClassificationStandardizerPrediction[T](
    basePrediction: PredictionResult[T],
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerPrediction[T] {

  override def getExpected(): Seq[T] = basePrediction.getExpected()

  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] =
    basePrediction.getUncertainty(includeNoise)

  override def getGradient(): Option[Seq[Vector[Double]]] = {
    basePrediction.getGradient().map { gradients =>
      gradients.map { g =>
        g.zip(inputTrans).map {
          case (y, inputStandardization) =>
            // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
            val inputScale = inputStandardization.map(_.scale).getOrElse(1.0)
            y / inputScale
        }
      }
    }
  }
}
