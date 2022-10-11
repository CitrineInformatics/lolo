package io.citrine.lolo.transformers

import io.citrine.lolo._
import io.citrine.random.Random

case class Standardization(shift: Double, scale: Double) {
  require(scale > 0 && scale < Double.PositiveInfinity)

  def apply(x: Double): Double = (x - shift) / scale

  def invert(x: Double): Double = x * scale + shift

}

/**
  * Standardize the training data to zero mean and unit variance before feeding it into another learner
  *
  * This is particularly helpful for regularized methods, like ridge regression, where
  * the relative scale of the features and labels is important.
  *
  * Created by maxhutch on 2/19/17.
  */
case class Standardizer[T](baseLearner: Learner[T]) extends Learner[T] {

  /**
    * Create affine transformations for continuous features and labels; pass data through to learner
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]],
      rng: Random
  ): StandardizerTrainingResult[T] = {
    val inputTrans = Standardizer.getMultiStandardization(trainingData.map(_._1))
    val outputTrans = trainingData.head._2 match {
      case _: Double => Some(Standardizer.getStandardization(trainingData.map(_._2.asInstanceOf[Double])))
      case _         => None
    }

    val (inputs, labels) = trainingData.unzip
    val standardInputs = Standardizer.applyStandardization(inputs, inputTrans)
    val standardLabels = Standardizer.applyStandardization(labels, outputTrans)
    val standardTrainingData = standardInputs.zip(standardLabels)

    val baseTrainingResult = baseLearner.train(standardTrainingData, weights, rng)
    new StandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

class MultiTaskStandardizer(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  /**
    * Train a model
    *
    * @param trainingData  to train on
    * @param weights for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a sequence of training results, one for each label
    */
  override def train(
      trainingData: Seq[(Vector[Any], Vector[Any])],
      weights: Option[Seq[Double]],
      rng: Random
  ): MultiTaskStandardizerTrainingResult = {
    val (inputs, labels) = trainingData.unzip
    val labelsTransposed = labels.transpose.toVector
    val repOutput = labels.head
    val inputTrans = Standardizer.getMultiStandardization(inputs)
    val outputTrans: Seq[Option[Standardization]] = repOutput.indices.map { i =>
      if (repOutput(i) != null && repOutput(i).isInstanceOf[Double]) {
        val labelSeq = labelsTransposed(i)
        Some(Standardizer.getStandardization(labelSeq.asInstanceOf[Seq[Double]].filterNot(_.isNaN())))
      } else {
        None
      }
    }
    val standardInputs = Standardizer.applyStandardization(inputs, inputTrans)
    val standardLabels = labelsTransposed
      .zip(outputTrans)
      .map {
        case (labelSeq, trans) =>
          Standardizer.applyStandardization(labelSeq, trans).toVector
      }
      .transpose

    val baseTrainingResult = baseLearner.train(standardInputs.zip(standardLabels), weights, rng)
    new MultiTaskStandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param outputTrans optional transformation (rescale, offset) of output label
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
class StandardizerTrainingResult[T](
    baseTrainingResult: TrainingResult[T],
    outputTrans: Option[Standardization],
    inputTrans: Seq[Option[Standardization]]
) extends TrainingResult[T] {

  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[T] =
    new StandardizerModel(baseTrainingResult.getModel(), outputTrans, inputTrans)

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], T, T)]] = {
    baseTrainingResult.getPredictedVsActual() match {
      case None => None
      case Some(predictedVsActual) =>
        Some(
          Standardizer
            .invertStandardization(predictedVsActual.map(_._1), inputTrans)
            .lazyZip(Standardizer.invertStandardization(predictedVsActual.map(_._2), outputTrans))
            .lazyZip(Standardizer.invertStandardization(predictedVsActual.map(_._3), outputTrans))
            .toSeq
        )
    }
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult result of training on the standardized inputs and output
  * @param outputTrans optional sequence of transformation (rescale, offset) of output labels
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  */
class MultiTaskStandardizerTrainingResult(
    baseTrainingResult: MultiTaskTrainingResult,
    outputTrans: Seq[Option[Standardization]],
    inputTrans: Seq[Option[Standardization]]
) extends MultiTaskTrainingResult {

  override def getModel(): MultiTaskModel = new ParallelModels(getModels(), baseTrainingResult.getModel().getRealLabels)

  override def getModels(): Seq[Model[Any]] =
    baseTrainingResult.getModels().zipWithIndex.map {
      case (model, i) =>
        new StandardizerModel(model, outputTrans(i), inputTrans)
    }

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] = {
    baseTrainingResult.getPredictedVsActual() match {
      case None => None
      case Some(predictedVsActual) =>
        Some(
          Standardizer
            .invertStandardization(predictedVsActual.map(_._1), inputTrans)
            .lazyZip(Standardizer.invertStandardizationOption(predictedVsActual.map(_._2), outputTrans))
            .lazyZip(Standardizer.invertStandardizationOption(predictedVsActual.map(_._3), outputTrans))
            .toSeq
        )
    }
  }
}

/**
  * Model that wraps the base model next to the transformations
  *
  * @param baseModel model trained on the standardized inputs and outputs
  * @param outputTrans optional transformation (rescale, offset) of output label
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  * @tparam T type of prediction
  */
class StandardizerModel[T](
    baseModel: Model[T],
    outputTrans: Option[Standardization],
    inputTrans: Seq[Option[Standardization]]
) extends Model[T] {

  /**
    * Standardize the inputs and then apply the base model
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): StandardizerPrediction[T] = {
    val standardInputs = Standardizer.applyStandardization(inputs, inputTrans)
    new StandardizerPrediction(baseModel.transform(standardInputs), outputTrans, inputTrans)
  }
}

/**
  * Prediction that wraps the base prediction next to the transformation
  *
  * @param baseResult result of applying model to standardized inputs
  * @param outputTrans optional transformation (rescale, offset) of output prediction
  * @param inputTrans sequence of optional transformations (rescale, offset) of inputs
  * @tparam T type of prediction
  */
class StandardizerPrediction[T](
    baseResult: PredictionResult[T],
    outputTrans: Option[Standardization],
    inputTrans: Seq[Option[Standardization]]
) extends PredictionResult[T] {

  /**
    * Get the expected values for this prediction
    *
    * Just reverse any transformation that was applied to the labels
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = {
    baseResult
      .getExpected()
      .map {
        case x: Double if outputTrans.isDefined => outputTrans.get.invert(x)
        case x: Any                             => x
      }
      .asInstanceOf[Seq[T]]
  }

  /**
    * Get the uncertainty of the prediction
    *
    * This is un-standardized by rescaling by the label's variance
    *
    * @return uncertainty of each prediction
    */
  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
    baseResult.getUncertainty(includeNoise) match {
      case Some(x) if outputTrans.isDefined => Some(x.map(_.asInstanceOf[Double] * outputRescale))
      case x: Any                           => x
    }
  }

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * This is un-stanardized by rescaling by the label's variance divided by the feature transformation's variance
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient() match {
      case None => None
      case Some(gradients) =>
        Some(
          gradients.map(g =>
            g.zip(inputTrans).map {
              // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
              case (y, Some(inputStandardization)) => y * outputRescale / inputStandardization.scale
              // Otherwise, just rescale by the output transformer
              case (y, None) => y * outputRescale
            }
          )
        )
    }
  }

  val outputRescale: Double = outputTrans.map(_.scale).getOrElse(1.0)
}

/**
  * Utilities to compute and apply standarizations
  */
object Standardizer {

  /**
    * The standardizations are just shifts and rescale.  The shift is by the mean and the re-scale is by the variance
    *
    * @param values to get a standardizer for
    * @return (shift, rescaling)
    */
  def getStandardization(values: Seq[Double]): Standardization = {
    val mean = values.sum / values.size
    val scale = Math.sqrt(values.map(v => Math.pow(v - mean, 2)).sum / values.size)

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
    * This has a different name because the jvm erases the inner type of Seq[T]
    *
    * @param values sequence of vectors to be standardized
    * @return sequence of standardization, each as an option
    */
  def getMultiStandardization(values: Seq[Vector[Any]]): Seq[Option[Standardization]] = {
    val rep = values.head
    rep.indices.map { i =>
      rep(i) match {
        case _: Double => Some(getStandardization(values.map(r => r(i).asInstanceOf[Double])))
        case _: Any    => None
      }
    }
  }

  /**
    * Apply the standardization to a sequence of values, which should result in output with zero mean and unit variance
    *
    * @param input to standardize
    * @param trans transformation to apply
    * @return sequence of standardized values
    */
  def applyStandardization[T](input: Seq[T], trans: Option[Standardization]): Seq[T] = {
    trans
      .map { t =>
        input.map {
          case x: Double => t.apply(x).asInstanceOf[T]
          case x         => x
        }
      }
      .getOrElse(input)
  }

  /**
    * Invert the standardization on a sequence of values
    *
    * @param input to invert the standardization
    * @param trans transformation to un-apply
    * @return sequence of restored values
    */
  def invertStandardization[T](input: Seq[T], trans: Option[Standardization]): Seq[T] = {
    trans
      .map { t =>
        input.map {
          case x: Double => t.invert(x).asInstanceOf[T]
          case x         => x
        }
      }
      .getOrElse(input)
  }

  /**
    * Apply the standardizations to vectors, which should result in an output with zero mean and unit variance
    *
    * @param input to standardize
    * @param trans transformtions to apply.  None means no transformation
    * @return sequence of standardized vectors
    */
  def applyStandardization(input: Seq[Vector[Any]], trans: Seq[Option[Standardization]]): Seq[Vector[Any]] = {
    input.map { r =>
      r.zip(trans).map {
        case (x: Double, Some(t)) => t.apply(x)
        case (x: Any, _)          => x
      }
    }
  }

  /**
    * Invert the standardizations on vectors.
    *
    * @param input  to invert the standardization
    * @param trans  transformations to un-apply. None means no transformation
    * @return       sequence of restored vectors
    */
  def invertStandardization(input: Seq[Vector[Any]], trans: Seq[Option[Standardization]]): Seq[Vector[Any]] = {
    input.map { r =>
      r.zip(trans).map {
        case (x: Double, Some(t)) => t.invert(x)
        case (x: Any, _)          => x
      }
    }
  }

  /**
    * Invert the standardizations on sequences of optional values.
    *
    * @param input  to invert the standardization, if the value is defined
    * @param trans  transformations to un-apply. None means no transformation
    * @return       sequence of restored vectors
    */
  def invertStandardizationOption(
      input: Seq[Vector[Option[Any]]],
      trans: Seq[Option[Standardization]]
  ): Seq[Vector[Option[Any]]] = {
    input.map { r =>
      r.zip(trans).map {
        case (Some(x: Double), Some(t)) => Some(t.invert(x))
        case (x: Any, _)                => x
      }
    }
  }
}
