package io.citrine.lolo.transformers

import io.citrine.lolo._

/**
  * Standardize the training data to zero mean and unit variance before feeding it into another learner
  *
  * This is particularly helpful for regularized methods, like ridge regression, where
  * the relative scale of the features and labels is important.
  *
  * Created by maxhutch on 2/19/17.
  */
case class Standardizer(baseLearner: Learner) extends Learner {

  /**
    * Create affine transformations for continuous features and labels; pass data through to learner
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return training result containing a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): StandardizerTrainingResult = {
    val inputTrans = Standardizer.getMultiStandardization(trainingData.map(_._1))
    val outputTrans = trainingData.head._2 match {
      case _: Double => Some(Standardizer.getStandardization(trainingData.map(_._2.asInstanceOf[Double])))
      case _: Any => None
    }

    val (inputs, labels) = trainingData.unzip
    val standardTrainingData = Standardizer.applyStandardization(inputs, inputTrans).zip(Standardizer.applyStandardization(labels, outputTrans))
    val baseTrainingResult = baseLearner.train(standardTrainingData, weights)

    new StandardizerTrainingResult(baseTrainingResult, outputTrans, inputTrans)
  }
}

class MultiTaskStandardizer(baseLearner: MultiTaskLearner) extends MultiTaskLearner {

  /**
    * Train a model
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return a sequence of training results, one for each label
    */
  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]]): MultiTaskTrainingResult = {
    val inputTrans = Standardizer.getMultiStandardization(inputs)
    val outputTrans: Seq[Option[(Double, Double)]] = labels.map { labelSeq =>
      if (labelSeq.head != null && labelSeq.head.isInstanceOf[Double]) {
        Some(Standardizer.getStandardization(labelSeq.asInstanceOf[Seq[Double]].filterNot(_.isNaN())))
      } else {
        None
      }
    }
    val standardInputs = Standardizer.applyStandardization(inputs, inputTrans)
    val standardLabels = labels.zip(outputTrans).map { case (labelSeq, trans) =>
      Standardizer.applyStandardization(labelSeq, trans)
    }

    val baseTrainingResult = baseLearner.train(standardInputs, standardLabels, weights)
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
class StandardizerTrainingResult(
                                  baseTrainingResult: TrainingResult,
                                  outputTrans: Option[(Double, Double)],
                                  inputTrans: Seq[Option[(Double, Double)]]
                                ) extends TrainingResult {
  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[PredictionResult[Any]] = new StandardizerModel(baseTrainingResult.getModel(), outputTrans, inputTrans)

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()
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
                                           outputTrans: Seq[Option[(Double, Double)]],
                                           inputTrans: Seq[Option[(Double, Double)]]
                                         ) extends MultiTaskTrainingResult {
  override def getModel(): MultiTaskModel = new ParallelModels(getModels(), baseTrainingResult.getModel().getRealLabels)

  override def getModels(): Seq[Model[PredictionResult[Any]]] = baseTrainingResult.getModels().zipWithIndex.map { case (model, i) =>
    new StandardizerModel(model, outputTrans(i), inputTrans)
  }

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()

  // TODO: extract predictedVsActual from baseTrainingResult and apply standardization
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
                            baseModel: Model[PredictionResult[T]],
                            outputTrans: Option[(Double, Double)],
                            inputTrans: Seq[Option[(Double, Double)]]
                          ) extends Model[PredictionResult[T]] {

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
                                 outputTrans: Option[(Double, Double)],
                                 inputTrans: Seq[Option[(Double, Double)]]
                               ) extends PredictionResult[T] {
  /**
    * Get the expected values for this prediction
    *
    * Just reverse any transformation that was applied to the labels
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[T] = {
    baseResult.getExpected().map {
      case x: Double => x * rescale + intercept
      case x: Any => x
    }.asInstanceOf[Seq[T]]
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
      case Some(x) if outputTrans.isDefined => Some(x.map(_.asInstanceOf[Double] * rescale))
      case x: Any => x
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
      case Some(x) =>
        Some(x.map(g => g.zip(inputTrans).map {
          case (0.0, Some((_, Double.PositiveInfinity))) => 0
          // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
          case (y: Double, Some((_, m))) => y * rescale * m
          // Otherwise, just rescale by the output transformer
          case (y, None) => y * rescale
        }))
    }
  }

  val rescale = 1.0 / outputTrans.map(_._2).getOrElse(1.0)
  val intercept = outputTrans.map(_._1).getOrElse(1.0)
}

/**
  * Utilities to compute and apply standarizations
  */
object Standardizer {

  /**
    * The standardizations are just shifts and rescale.  The shift is by the mean and the re-scale is by the variance
    *
    * @param values to get a standardizer for
    * @return
    */
  def getStandardization(values: Seq[Double]): (Double, Double) = {
    val mean = values.sum / values.size
    val scale = Math.sqrt(values.map(v => Math.pow(v - mean, 2)).sum / values.size)

    // If there is zero variance, then the scaling doesn't matter; default to 1.0
    if (scale > 0) {
      (mean, 1.0 / scale)
    } else {
      (mean, 1.0)
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
  def getMultiStandardization(values: Seq[Vector[Any]]): Seq[Option[(Double, Double)]] = {
    val rep = values.head
    rep.indices.map { i =>
      rep(i) match {
        case _: Double =>
          Some(getStandardization(values.map(r => r(i).asInstanceOf[Double])))
        case _: Any => None
      }
    }
  }

  /**
    * Apply the standardization to vectors, which should result in an output with zero mean and unit variance
    *
    * @param input to standardize
    * @param trans transformtions to apply.  None means no transformation
    * @return sequence of standardized vectors
    */
  def applyStandardization(input: Seq[Vector[Any]], trans: Seq[Option[(Double, Double)]]): Seq[Vector[Any]] = {
    input.map { r =>
      r.zip(trans).map {
        case (x: Double, Some(t)) => (x - t._1) * t._2
        case (x: Any, _) => x
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
  def applyStandardization(input: Seq[Any], trans: Option[(Double, Double)]): Seq[Any] = {
    if (trans.isEmpty) return input

    input.asInstanceOf[Seq[Double]].map { r =>
      (r - trans.get._1) * trans.get._2
    }
  }
}
