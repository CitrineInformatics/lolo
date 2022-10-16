package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo.stats.StatsUtils
import io.citrine.lolo.{
  Learner,
  Model,
  MultiTaskLearner,
  MultiTaskModel,
  MultiTaskModelPredictionResult,
  MultiTaskTrainingResult,
  ParallelModelsPredictionResult,
  PredictionResult
}
import io.citrine.random.Random
import io.citrine.lolo.stats.metrics.{ClassificationMetrics, RegressionMetrics}

import scala.collection.parallel.immutable.ParSeq
import scala.collection.parallel.CollectionConverters._

/**
  * Create an ensemble of multi-task models.
  *
  * @param method                 learner to train each model in the ensemble
  * @param numBags                number of models in the ensemble
  * @param useJackknife           whether to enable jackknife uncertainty estimate
  * @param uncertaintyCalibration whether to empirically recalibrate the predicted uncertainties
  * @param biasLearner            learner to use for estimating bias
  */
case class MultiTaskBagger(
    method: MultiTaskLearner,
    numBags: Int = -1,
    useJackknife: Boolean = true,
    uncertaintyCalibration: Boolean = true,
    biasLearner: Option[Learner[Double]] = None
) extends MultiTaskLearner {

  override def train(
      trainingData: Seq[(Vector[Any], Vector[Any])],
      weights: Option[Seq[Double]],
      rng: Random
  ): MultiTaskBaggedTrainingResult = {
    val (inputs, labels) = trainingData.unzip
    val numInputs = inputs.head.length
    val numOutputs = labels.head.length
    /* Make sure the training data are the same size */
    assert(inputs.forall(numInputs == _.size))
    assert(labels.forall(numOutputs == _.size))
    assert(
      trainingData.size >= Bagger.minimumTrainingSize,
      s"We need to have at least ${Bagger.minimumTrainingSize} rows, only ${trainingData.size} given"
    )
    (0 until numOutputs).foreach { i =>
      val numOutputValues = labels.count(row => validOutput(row(i)))
      assert(
        numOutputValues >= Bagger.minimumOutputCount,
        s"There must be at least ${Bagger.minimumOutputCount} data points for each output, but output $i only had $numOutputValues values."
      )
    }

    // if numBags is non-positive, set # bags = # inputs
    val actualBags = if (numBags > 0) numBags else trainingData.size

    // Compute the number of instances of each training row in each training sample
    val randBasis = StatsUtils.breezeRandBasis(rng)
    val dist = new Poisson(1.0)(randBasis)
    val Nib: Vector[Vector[Int]] = Iterator
      .continually(Vector.fill(trainingData.size)(dist.draw()))
      .filter { suggestedCounts =>
        val allOutputsRepresented = (0 until numOutputs).forall(i =>
          labels.zip(suggestedCounts).exists { case (row, count) => validOutput(row(i)) && count > 0 }
        )
        val minNonzeroWeights = suggestedCounts.count(_ > 0) >= Bagger.minimumNonzeroWeightSize
        allOutputsRepresented && minNonzeroWeights
      }
      .take(actualBags)
      .toVector
    val weightsActual = weights.getOrElse(Seq.fill(trainingData.size)(1.0))

    val indices = Nib.indices.toVector
    val (models, importances) =
      rng
        .zip(indices)
        .par
        .map {
          case (thisRng, i) =>
            val sampleWeights = Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)
            val meta = method.train(trainingData, Some(sampleWeights), thisRng)
            (meta.getModel(), meta.getFeatureImportance())
        }
        .unzip

    val averageImportance: Option[Vector[Double]] = importances
      .reduce(Bagger.combineImportance)
      .map(_.map(_ / importances.size))

    // Get bias model and rescale ratio for each label
    val (biasModels, rescaleRatios) = Seq
      .tabulate(numOutputs) { i =>
        val isRegression = models.head.getRealLabels(i)
        if (isRegression) {
          val thisLabelModels = models.map(_.getModels(i).asInstanceOf[Model[Double]])
          val thisTrainingData = trainingData.map {
            case (inputs, outputs) => (inputs, outputs(i).asInstanceOf[Double])
          }
          val helper = BaggerHelper(thisLabelModels, thisTrainingData, Nib, useJackknife, uncertaintyCalibration)
          val biasModel = biasLearner.collect {
            case learner if helper.oobErrors.nonEmpty =>
              learner.train(helper.biasTraining, rng = rng).getModel()
          }
          (biasModel, Some(helper.rescaleRatio))
        } else {
          (None, None)
        }
      }
      .unzip

    new MultiTaskBaggedTrainingResult(
      models = models,
      Nib = Nib,
      featureImportance = averageImportance,
      trainingData = trainingData,
      biasModels = biasModels,
      rescaleRatios = rescaleRatios
    )
  }

  /** Flag NaNs and nulls. */
  private def validOutput(x: Any): Boolean = {
    Option(x) match {
      case Some(x: Double) => !x.isNaN
      case Some(_: Any)    => true
      case None            => false
    }
  }
}

/**
  * The result of training a bagger on a multi-label combined model.
  *
  * @param models            sequence of multi-models, one for each bag
  * @param featureImportance importance of input features
  * @param Nib               matrix representing number of times each training datum appears in each bag
  * @param trainingData      multi-label training data
  * @param biasModels        sequence of optional bias-correction models, one for each label
  * @param rescaleRatios     sequence of uncertainty calibration ratios for each label
  */
class MultiTaskBaggedTrainingResult(
    models: ParSeq[MultiTaskModel],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[(Vector[Any], Vector[Any])],
    featureImportance: Option[Vector[Double]],
    biasModels: Seq[Option[Model[Double]]],
    rescaleRatios: Seq[Option[Double]]
) extends MultiTaskTrainingResult {

  lazy val model = new MultiTaskBaggedModel(models, Nib, biasModels, rescaleRatios)

  // Each entry is a tuple, (feature vector, seq of predicted labels, seq of actual labels).
  // The labels are of type Option[Any] because a given training datum might not have a value for every single label.
  // If the actual value for a label is None, then the corresponding prediction is recorded as None. The model could generate
  // a prediction, but that's not useful in this context, since the point is to compare predictions with ground-truth values.
  lazy val predictedVsActual: Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])] =
    trainingData.zip(Nib.transpose).flatMap {
      case ((features, labels), nb) =>
        // Bagged models that were not trained on this input
        val oob = models.zip(nb).filter(_._2 == 0).map(_._1)
        if (oob.isEmpty) {
          Seq()
        } else {
          // "Average" the predictions on each label over the out-of-bag models
          val oobPredictions = oob.map(_.transform(Seq(features)).getExpected().head)
          val predicted = oobPredictions.toVector.transpose.zipWithIndex.map {
            case (predictions, labelIndex) if models.head.getRealLabels(labelIndex) =>
              predictions.asInstanceOf[Seq[Double]].sum / predictions.size
            case (predictions, _) => predictions.groupBy(identity).maxBy(_._2.size)._1
          }
          // Remove predictions for which the label was not specified
          val (optionLabels, optionPredicted) = labels
            .zip(predicted)
            .map {
              case (l, _) if l == null || (l.isInstanceOf[Double] && l.asInstanceOf[Double].isNaN) => (None, None)
              case (l, p)                                                                          => (Some(l), Some(p))
            }
            .unzip
          Seq((features, optionPredicted, optionLabels))
        }
    }

  lazy val loss: Double = {
    val allInputs = predictedVsActual.map(_._1)
    val allPredicted: Seq[Seq[Option[Any]]] = predictedVsActual.map(_._2).transpose
    val allActual: Seq[Seq[Option[Any]]] = predictedVsActual.map(_._3).transpose
    allPredicted.lazyZip(allActual).lazyZip(models.head.getRealLabels).map {
      case (labelPredicted, labelActual, isReal) =>
        // Construct predicted-vs-actual for just this label, only keeping entries for which both predicted and actual are defined
        val pva = allInputs.lazyZip(labelPredicted).lazyZip(labelActual).flatMap {
          case (input, Some(p), Some(a)) => Some((input, p, a))
          case _                         => None
        }
        if (isReal) {
          RegressionMetrics.RMSE(pva.asInstanceOf[Seq[(Vector[Any], Double, Double)]])
        } else {
          ClassificationMetrics.loss(pva)
        }
    }
  }.sum

  override def getFeatureImportance(): Option[Vector[Double]] = featureImportance

  override def getModel(): MultiTaskBaggedModel = model

  override def getModels(): Seq[Model[Any]] = {
    val realLabels: Seq[Boolean] = models.head.getRealLabels
    realLabels.zipWithIndex.map {
      case (isReal: Boolean, i: Int) =>
        val thisLabelModels = models.map(_.getModels(i))
        if (isReal) {
          new BaggedRegressionModel(
            thisLabelModels.asInstanceOf[ParSeq[Model[Double]]],
            Nib = Nib,
            rescale = rescaleRatios(i).get,
            biasModel = biasModels(i)
          )
        } else {
          new BaggedClassificationModel(thisLabelModels, Nib)
        }
    }
  }

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] =
    Some(predictedVsActual)

  override def getLoss(): Option[Double] = {
    if (predictedVsActual.nonEmpty) Some(loss) else None
  }
}

/**
  * Container holding a parallel sequence of models, each of which predicts on multiple labels.
  *
  * @param models       sequence of multi-models, one for each bag
  * @param Nib          matrix representing number of times each training datum appears in each bag
  * @param biasModels   sequence of optional bias-correction models, one for each label
  * @param rescaleRatios  sequence of uncertainty calibration ratios for each label
  */
class MultiTaskBaggedModel(
    models: ParSeq[MultiTaskModel],
    Nib: Vector[Vector[Int]],
    biasModels: Seq[Option[Model[Double]]],
    rescaleRatios: Seq[Option[Double]]
) extends MultiTaskModel {

  override val numLabels: Int = models.head.numLabels

  lazy val groupedModels: Vector[BaggedModel[Any]] = Vector.tabulate(numLabels) { i =>
    val thisLabelsModels = models.map(_.getModels(i))
    if (getRealLabels(i)) {
      new BaggedRegressionModel(
        thisLabelsModels.asInstanceOf[ParSeq[Model[Double]]],
        Nib = Nib,
        rescale = rescaleRatios(i).get,
        biasModel = biasModels(i)
      )
    } else {
      new BaggedClassificationModel(thisLabelsModels, Nib)
    }
  }

  override def transform(inputs: Seq[Vector[Any]]): MultiTaskBaggedResult =
    MultiTaskBaggedResult(groupedModels.map(_.transform(inputs)), getRealLabels, Nib)

  override def getRealLabels: Seq[Boolean] = models.head.getRealLabels

  override def getModels: Seq[BaggedModel[Any]] = groupedModels
}

/**
  * Container with model-wise predictions for each label and the machinery to compute (co)variance.
  *
  * @param baggedPredictions  bagged prediction results for each label
  * @param realLabels         a boolean sequence indicating which labels are real-valued
  * @param NibIn              the sampling matrix as (# bags) x (# training)
  */
case class MultiTaskBaggedResult(
                                  baggedPredictions: Vector[BaggedPrediction[Any]],
                                  realLabels: Seq[Boolean],
                                  NibIn: Vector[Vector[Int]]
) extends BaggedPrediction[Vector[Any]]
    with MultiTaskModelPredictionResult {

  /* transpose to be (# training) x (# models) */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose

  override def numPredictions: Int = baggedPredictions.head.numPredictions

  override def getExpected(): Seq[Vector[Any]] = baggedPredictions.map(_.getExpected()).transpose

  override def predictions: Seq[PredictionResult[Vector[Any]]] =
    baggedPredictions
      .map(_.predictions.map(_.getExpected()))
      .transpose
      .map(x => new ParallelModelsPredictionResult(x.transpose))

  // For each prediction, the uncertainty is a sequence of entries for each label. Missing uncertainty values are reported as NaN
  override def getUncertainty(observational: Boolean = true): Option[Seq[Seq[Any]]] = {
    Some(baggedPredictions.map { predictionResult =>
      predictionResult.getUncertainty(observational) match {
        case Some(value) => value
        case None        => Seq.fill(numPredictions)(Double.NaN)
      }
    }.transpose)
  }

  override def getUncertaintyCorrelation(i: Int, j: Int, observational: Boolean = true): Option[Seq[Double]] = {
    (realLabels(i), realLabels(j)) match {
      case (true, true) if i == j => Some(Seq.fill(numPredictions)(1.0))
      case (true, true) =>
        if (observational) {
          Some(uncertaintyCorrelationObservational(i, j))
        } else {
          Some(uncertaintyCorrelationMean)
        }
      case _: Any => None
    }
  }

  /**
    * The uncertainty correlation of the observational distribution is the correlation coefficient
    * calculated over the bootstrap ensemble predictions.
    */
  private def uncertaintyCorrelationObservational(i: Int, j: Int): Seq[Double] = {
    // make (# predictions) x (# bags) prediction matrices for each label
    val baggedPredictionsI =
      baggedPredictions(i).predictions.map(_.getExpected()).transpose.asInstanceOf[Seq[Seq[Double]]]
    val baggedPredictionsJ =
      baggedPredictions(j).predictions.map(_.getExpected()).transpose.asInstanceOf[Seq[Seq[Double]]]
    baggedPredictionsI.zip(baggedPredictionsJ).map {
      case (bagsI, bagsJ) =>
        StatsUtils.correlation(bagsI, bagsJ)
    }
  }

  /**
    * The uncertainty correlation of the mean distribution is 0.0. In theory it should be estimated using the jackknife,
    * but in practice the jackknife performs poorly when estimating covariance, so we default to the trivial implementation for now.
    */
  private def uncertaintyCorrelationMean: Seq[Double] = Seq.fill(numPredictions)(0.0)
}
