package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo.stats.StatsUtils.breezeRandBasis
import io.citrine.lolo.{Learner, Model, MultiTaskLearner, MultiTaskModel, MultiTaskTrainingResult, PredictionResult}
import io.citrine.random.Random
import io.citrine.lolo.stats.metrics.{ClassificationMetrics, RegressionMetrics}
import io.citrine.lolo.util.Async

import scala.collection.parallel.immutable.ParSeq
import scala.collection.parallel.CollectionConverters.IterableIsParallelizable

/**
  * Create an ensemble of multi-task models
  *
  * @param method                 learner to train each model in the ensemble
  * @param numBags                number of models in the ensemble
  * @param useJackknife           whether to enable jackknife uncertainty estimate
  * @param biasLearner            learner to use for estimating bias
  * @param uncertaintyCalibration whether to empirically recalibrate the predicted uncertainties
  */
case class MultiTaskBagger(
    method: MultiTaskLearner,
    numBags: Int = -1,
    useJackknife: Boolean = true,
    biasLearner: Option[Learner] = None,
    uncertaintyCalibration: Boolean = true
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
    val randBasis = breezeRandBasis(rng)
    val dist = new Poisson(1.0)(randBasis)
    val Nib: Vector[Vector[Int]] = Iterator
      .continually(Vector.fill(trainingData.size)(dist.draw))
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
    val (models: ParSeq[MultiTaskModel], importances: ParSeq[Option[Vector[Double]]]) =
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
      .reduce(combineImportance)
      .map(_.map(_ / importances.size))

    // Get bias model and rescale ratio for each label
    val (biasModels, ratios) = Seq
      .tabulate(numOutputs) { i =>
        val thisLabelModels: ParSeq[Model[PredictionResult[Any]]] = models.map(_.getModels(i))
        val isRegression = models.head.getRealLabels(i)
        val thisTrainingData = trainingData.map { case (inputs, outputs) => (inputs, outputs(i)) }
        val helper = BaggerHelper(thisLabelModels, thisTrainingData, Nib, useJackknife, uncertaintyCalibration)
        val biasModel = if (biasLearner.isDefined && isRegression) {
          Async.canStop()
          Some(
            biasLearner.get
              .train(helper.biasTraining, rng = rng)
              .getModel()
              .asInstanceOf[Model[PredictionResult[Double]]]
          )
        } else None
        (biasModel, helper.rescaleRatio)
      }
      .unzip

    new MultiTaskBaggedTrainingResult(
      models = models,
      featureImportance = averageImportance,
      Nib = Nib,
      trainingData = trainingData,
      useJackknife = useJackknife,
      biasModels = biasModels,
      rescaleRatios = ratios
    )
  }

  /** Combine two optional feature importance vectors. */
  def combineImportance(v1: Option[Vector[Double]], v2: Option[Vector[Double]]): Option[Vector[Double]] = {
    (v1, v2) match {
      case (Some(v1: Vector[Double]), Some(v2: Vector[Double])) => Some(v1.zip(v2).map(p => p._1 + p._2))
      case _                                                    => None
    }
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
  * @param models             sequence of multi-models, one for each bag
  * @param featureImportance  importance of input features
  * @param Nib                matrix representing number of times each training datum appears in each bag
  * @param trainingData       multi-label training data
  * @param useJackknife       whether to enable jackknife uncertainty estimate
  * @param biasModels         sequence of optional bias-correction models, one for each label
  * @param rescaleRatios      sequence of uncertainty calibration ratios for each label
  */
class MultiTaskBaggedTrainingResult(
    models: ParSeq[MultiTaskModel],
    featureImportance: Option[Vector[Double]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[(Vector[Any], Seq[Any])],
    useJackknife: Boolean,
    biasModels: Seq[Option[Model[PredictionResult[Double]]]],
    rescaleRatios: Seq[Double]
) extends MultiTaskTrainingResult {

  lazy val model = new MultiTaskBaggedModel(models, Nib, useJackknife, biasModels, rescaleRatios)

  // Each entry is a tuple, (feature vector, seq of predicted labels, seq of actual labels).
  // The labels are of type Option[Any] because a given training datum might not have a value for every single label.
  // If the actual value for a label is None, then the corresponding prediction is recorded as None. The model could generate
  // a prediction, but that's not useful in this context, since the point is to compare predictions with ground-truth values.
  lazy val predictedVsActual: Seq[(Vector[Any], Seq[Option[Any]], Seq[Option[Any]])] =
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

  lazy val loss = {
    val allInputs = predictedVsActual.map(_._1)
    val allPredicted: Seq[Seq[Option[Any]]] = predictedVsActual.map(_._2).transpose
    val allActual: Seq[Seq[Option[Any]]] = predictedVsActual.map(_._3).transpose
    (allPredicted, allActual, models.head.getRealLabels).zipped.map {
      case (labelPredicted, labelActual, isReal) =>
        // Construct predicted-vs-actual for just this label, only keeping entries for which both predicted and actual are defined
        val pva = (allInputs, labelPredicted, labelActual).zipped.flatMap {
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

  override def getModels(): Seq[Model[PredictionResult[Any]]] = {
    val realLabels: Seq[Boolean] = models.head.getRealLabels
    realLabels.zipWithIndex.map {
      case (isReal: Boolean, i: Int) =>
        val thisLabelModels = models.map(_.getModels(i))
        if (isReal) {
          new BaggedModel[Double](
            thisLabelModels.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]],
            Nib,
            useJackknife,
            biasModels(i),
            rescaleRatios(i)
          )
        } else {
          new BaggedModel[Any](
            thisLabelModels.asInstanceOf[ParSeq[Model[PredictionResult[Any]]]],
            Nib,
            useJackknife,
            biasModels(i),
            rescaleRatios(i)
          )
        }
    }
  }

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Seq[Option[Any]], Seq[Option[Any]])]] =
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
  * @param useJackknife whether to enable jackknife uncertainty estimate
  * @param biasModels   sequence of optional bias-correction models, one for each label
  * @param rescaleRatios  sequence of uncertainty calibration ratios for each label
  */
class MultiTaskBaggedModel(
    models: ParSeq[MultiTaskModel],
    Nib: Vector[Vector[Int]],
    useJackknife: Boolean,
    biasModels: Seq[Option[Model[PredictionResult[Double]]]],
    rescaleRatios: Seq[Double]
) extends MultiTaskModel {

  lazy val groupedModels: Seq[BaggedModel[Any]] = Seq.tabulate(numLabels) { i =>
    val thisLabelsModels = models.map(_.getModels(i))
    if (getRealLabels(i)) {
      new BaggedModel[Double](
        thisLabelsModels.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]],
        Nib,
        useJackknife,
        biasModels(i),
        rescaleRatios(i)
      )
    } else {
      new BaggedModel(thisLabelsModels, Nib, useJackknife, biasModels(i), rescaleRatios(i))
    }
  }

  override def transform(inputs: Seq[Vector[Any]]) =
    MultiTaskBaggedResult(groupedModels.map(_.transform(inputs)), getRealLabels, Nib)

  override val numLabels: Int = models.head.numLabels

  override def getRealLabels: Seq[Boolean] = models.head.getRealLabels

  override def getModels: Seq[BaggedModel[Any]] = groupedModels
}
