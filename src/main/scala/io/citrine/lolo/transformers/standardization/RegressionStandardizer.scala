package io.citrine.lolo.transformers.standardization

import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}
import io.citrine.random.Random

case class RegressionStandardizer(baseLearner: Learner[Double]) extends Learner[Double] {

  /**
    * Create affine transformations for continuous features and labels; pass data through to the base learner.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Double)],
      weights: Option[Seq[Double]],
      rng: Random
  ): RegressionStandardizerTrainingResult = {
    val inputTrans = Standardization.buildMulti(trainingData.map(_._1))
    val outputTrans = Standardization.build(trainingData.map(_._2))

    val (inputs, labels) = trainingData.unzip
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardLabels = labels.map(outputTrans.apply)
    val standardTrainingData = standardInputs.zip(standardLabels)

    val baseTrainingResult = baseLearner.train(standardTrainingData, weights, rng)
    RegressionStandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param outputTrans optional transformation (rescale, offset) of output label
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
case class RegressionStandardizerTrainingResult(
    baseTrainingResult: TrainingResult[Double],
    outputTrans: Standardization,
    inputTrans: Map[Int, Standardization]
) extends TrainingResult[Double] {

  override def getModel(): RegressionStandardizerModel =
    RegressionStandardizerModel(baseTrainingResult.getModel(), outputTrans, inputTrans)

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Double, Double)]] = {
    baseTrainingResult.getPredictedVsActual().map { pva =>
      pva.map {
        case (inputs, pred, actual) =>
          (Standardization.invertMulti(inputs, inputTrans), outputTrans.invert(pred), outputTrans.invert(actual))
      }
    }
  }
}

case class RegressionStandardizerModel(
    baseModel: Model[Double],
    outputTrans: Standardization,
    inputTrans: Map[Int, Standardization]
) extends Model[Double] {

  override def transform(inputs: Seq[Vector[Any]]): RegressionStandardizerPrediction = {
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    RegressionStandardizerPrediction(baseModel.transform(standardInputs), outputTrans, inputTrans)
  }
}

case class RegressionStandardizerPrediction(
    baseResult: PredictionResult[Double],
    outputTrans: Standardization,
    inputTrans: Map[Int, Standardization]
) extends PredictionResult[Double] {

  override def getExpected(): Seq[Double] = baseResult.getExpected().map(outputTrans.invert)

  // TODO: A PredictionResult[Double] should always return a Option[Seq[Double]] for uncertainty
  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
    baseResult.getUncertainty(includeNoise) match {
      case Some(x) => Some(x.map(_.asInstanceOf[Double] * outputTrans.scale))
      case x       => x
    }
  }

  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient().map { gradients =>
      gradients.map { g =>
        g.zipWithIndex.map {
          case (y, idx) =>
            // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
            // Otherwise, just rescale by the output transformer
            val inputScale = inputTrans.get(idx).map(_.scale).getOrElse(1.0)
            val outputScale = outputTrans.scale
            y * outputScale / inputScale
        }
      }
    }
  }
}
