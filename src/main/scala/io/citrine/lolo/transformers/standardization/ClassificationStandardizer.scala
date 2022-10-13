package io.citrine.lolo.transformers.standardization

import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}
import io.citrine.random.Random

case class ClassificationStandardizer(baseLearner: Learner[Any]) extends Learner[Any] {

  /**
    * Create affine transformations for continuous features and labels; pass data through to the base learner.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): ClassificationStandardizerTrainingResult = {
    val inputTrans = Standardization.buildMulti(trainingData.map(_._1))

    val (inputs, labels) = trainingData.unzip
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    val standardTrainingData = standardInputs.zip(labels)

    val baseTrainingResult = baseLearner.train(standardTrainingData, weights, rng)
    ClassificationStandardizerTrainingResult(baseTrainingResult, inputTrans)
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
case class ClassificationStandardizerTrainingResult(
    baseTrainingResult: TrainingResult[Any],
    inputTrans: Seq[Option[Standardization]]
) extends TrainingResult[Any] {

  override def getModel(): ClassificationStandardizerModel =
    ClassificationStandardizerModel(baseTrainingResult.getModel(), inputTrans)

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Any, Any)]] = {
    baseTrainingResult.getPredictedVsActual().map { pva =>
      pva.map {
        case (inputs, pred, actual) => (Standardization.invertMulti(inputs, inputTrans), pred, actual)
      }
    }
  }
}

case class ClassificationStandardizerModel(
    baseModel: Model[Any],
    inputTrans: Seq[Option[Standardization]]
) extends Model[Any] {

  override def transform(inputs: Seq[Vector[Any]]): ClassificationStandardizerPrediction = {
    val standardInputs = inputs.map { input => Standardization.applyMulti(input, inputTrans) }
    ClassificationStandardizerPrediction(baseModel.transform(standardInputs), inputTrans)
  }
}

case class ClassificationStandardizerPrediction(
    baseResult: PredictionResult[Any],
    inputTrans: Seq[Option[Standardization]]
) extends PredictionResult[Any] {

  override def getExpected(): Seq[Any] = baseResult.getExpected()

  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = baseResult.getUncertainty(includeNoise)

  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient().map { gradients =>
      gradients.map { g =>
        g.zip(inputTrans).map {
          // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
          case (y, Some(inputStandardization)) => y / inputStandardization.scale
          case (y, _)                          => y
        }
      }
    }
  }
}
