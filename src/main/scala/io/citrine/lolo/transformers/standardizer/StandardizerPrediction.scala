package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.PredictionResult

trait StandardizerPrediction[+T] extends PredictionResult[T] {}

case class RegressionStandardizerPrediction(
    baseResult: PredictionResult[Double],
    outputTrans: Standardization,
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerPrediction[Double] {

  override def getExpected(): Seq[Double] = baseResult.getExpected().map(outputTrans.invert)

  // TODO: A PredictionResult[Double] should always return a Option[Seq[Double]] for uncertainty
  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
    baseResult.getUncertainty(includeNoise).map { x =>
      x.map(_.asInstanceOf[Double] * outputTrans.scale)
    }
  }

  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient().map { gradients =>
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
    baseResult: PredictionResult[T],
    inputTrans: Seq[Option[Standardization]]
) extends StandardizerPrediction[T] {

  override def getExpected(): Seq[T] = baseResult.getExpected()

  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = baseResult.getUncertainty(includeNoise)

  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient().map { gradients =>
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
