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
      case x: Double => Some(Standardizer.getStandardization(trainingData.map(_._2.asInstanceOf[Double])))
      case x: Any => None
    }

    val (inputs, labels) = trainingData.unzip
    val standardTrainingData = Standardizer.applyStandardization(inputs, inputTrans).zip(Standardizer.applyStandardization(labels, outputTrans))
    val baseTrainingResult = baseLearner.train(standardTrainingData, weights)

    new StandardizerTrainingResult(baseTrainingResult, Seq(outputTrans) ++ inputTrans)
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
  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]]): Seq[TrainingResult] = {
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

    baseTrainingResult.zip(outputTrans).map { case (base, trans) =>
      new StandardizerTrainingResult(base, Seq(trans) ++ inputTrans)
    }
  }
}

/**
  * Training result wrapping the base learner's training result next to the transformations
  *
  * @param baseTrainingResult
  * @param trans
  */
class StandardizerTrainingResult(
                                  baseTrainingResult: TrainingResult,
                                  trans: Seq[Option[(Double, Double)]]
                                ) extends TrainingResult {
  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[PredictionResult[Any]] = new StandardizerModel(baseTrainingResult.getModel(), trans)

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()
}

/**
  * Model that wrapps the base model next to the transformations
  *
  * @param baseModel
  * @param trans
  * @tparam T
  */
class StandardizerModel[T](baseModel: Model[PredictionResult[T]], trans: Seq[Option[(Double, Double)]]) extends Model[PredictionResult[T]] {

  /**
    * Standardize the inputs and then apply the base model
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): StandardizerPrediction[T] = {
    val standardInputs = Standardizer.applyStandardization(inputs, trans.tail)
    new StandardizerPrediction(baseModel.transform(standardInputs), trans)
  }
}

/**
  * Prediction that wraps the base prediction next to the transformation
  *
  * @param baseResult
  * @param trans
  * @tparam T
  */
class StandardizerPrediction[T](baseResult: PredictionResult[T], trans: Seq[Option[(Double, Double)]]) extends PredictionResult[T] {
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
      case Some(x) if trans.head.isDefined => Some(x.map(_.asInstanceOf[Double] * rescale))
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
        Some(x.map(g => g.zip(trans.tail).map {
          case (0.0, Some((_, Double.PositiveInfinity))) => 0
          // If there was a (linear) transformer used on that input, take the slope "m" and rescale by it
          case (y: Double, Some((_, m))) => y * rescale * m
          // Otherwise, just rescale by the output transformer
          case (y, None) => y * rescale
        }))
    }
  }

  val rescale = 1.0 / trans.head.map(_._2).getOrElse(1.0)
  val intercept = trans.head.map(_._1).getOrElse(1.0)
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
        case x: Double =>
          Some(getStandardization(values.map(r => r(i).asInstanceOf[Double])))
        case x: Any => None
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
