package io.citrine.lolo.transformers

import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

/**
  * Created by maxhutch on 2/19/17.
  */
class Standardizer(baseLearner: Learner) extends Learner() {
  override var hypers: Map[String, Any] = Map()

  override def setHypers(moreHypers: Map[String, Any]): this.type = {
    baseLearner.setHypers(moreHypers)
    super.setHypers(moreHypers)
  }

  /**
    * Train a model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return training result containing a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): StandardizerTrainingResult = {
    val rep = trainingData.head._1
    val inputTrans = Standardizer.getMultiStandardization(trainingData.map(_._1))
    val outputTrans = trainingData.head._2 match {
      case x: Double => Some(Standardizer.getStandardization(trainingData.map(_._2.asInstanceOf[Double])))
      case x: Any => None
    }

    val (inputs, labels) = trainingData.unzip
    val standardTrainingData = Standardizer.applyStandardization(inputs, inputTrans).zip(Standardizer.applyStandardization(labels, outputTrans))
    val baseTrainingResult = baseLearner.train(standardTrainingData)

    new StandardizerTrainingResult(baseTrainingResult, Seq(outputTrans) ++ inputTrans, hypers)
  }
}

class StandardizerTrainingResult(
                                  baseTrainingResult: TrainingResult,
                                  trans: Seq[Option[(Double, Double)]],
                                  hypers: Map[String, Any]
                                ) extends TrainingResult {
  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[PredictionResult[Any]] = new StandardizerModel(baseTrainingResult.getModel(), trans)

  /**
    * Get the hyperparameters used to train this model
    *
    * @return hypers set for model
    */
  override def getHypers(): Map[String, Any] = hypers

  override def getFeatureImportance(): Option[Vector[Double]] = baseTrainingResult.getFeatureImportance()
}

class StandardizerModel[T](baseModel: Model[PredictionResult[T]], trans: Seq[Option[(Double, Double)]]) extends Model[PredictionResult[T]] {
  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): StandardizerPrediction[T] = {
    val standardInputs = Standardizer.applyStandardization(inputs, trans.tail)
    new StandardizerPrediction(baseModel.transform(standardInputs), trans)
  }
}

class StandardizerPrediction[T](baseResult: PredictionResult[T], trans: Seq[Option[(Double, Double)]]) extends PredictionResult[T] {
  /**
    * Get the expected values for this prediction
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
    * For example, in regression this is sqrt(bias^2 + variance)
    * @return uncertainty of each prediction
    */
  override def getUncertainty(): Option[Seq[Any]] = {
    baseResult.getUncertainty() match {
      case None => None
      case Some(x: Seq[Double]) => Some(x.map(_ * rescale))
      case Some(x: Seq[Any]) => Some(x)
    }
  }


  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = {
    baseResult.getGradient() match {
      case None => None
      case Some(x) =>
        Some(x.map(g => g.zip(trans.tail).map {
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

object Standardizer {
  def getStandardization(values: Seq[Double]): (Double, Double) = {
    val mean = values.sum / values.size
    val scale = Math.sqrt(values.map(v => Math.pow(v - mean, 2)).sum / values.size)
    (mean, 1.0 / scale)
  }

  /**
    * Get standardization for multiple values in a vector.
    *
    * This has a different name because scala erases the inner type of Seq[T]
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

  def applyStandardization(input: Seq[Vector[Any]], trans: Seq[Option[(Double, Double)]]): Seq[Vector[Any]] = {
    input.map { r =>
      r.zip(trans).map {
        case (x: Double, Some(t)) => (x - t._1) * t._2
        case (x: Any, None) => x
      }
    }
  }

  def applyStandardization(input: Seq[Any], trans: Option[(Double, Double)]): Seq[Any] = {
    if (trans.isEmpty) return input

    input.asInstanceOf[Seq[Double]].map { r =>
      (r - trans.get._1) * trans.get._2
    }
  }
}
