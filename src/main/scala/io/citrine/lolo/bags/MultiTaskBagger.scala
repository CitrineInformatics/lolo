package io.citrine.lolo.bags

import breeze.stats.distributions.{Poisson, Rand, RandBasis}
import io.citrine.lolo._
import io.citrine.lolo.util.{Async, InterruptibleExecutionContext}

import scala.collection.parallel.ExecutionContextTaskSupport
import scala.collection.parallel.immutable.{ParRange, ParSeq}

/**
  * Create an ensemble of multi-task models
  *
  * @param method                 learner to train each model in the ensemble
  * @param numBags                number of models in the ensemble
  * @param useJackknife           whether to enable jackknife uncertainty estimate
  * @param biasLearner            learner to use for estimating bias
  * @param uncertaintyCalibration whether to empirically recalibrate the predicted uncertainties
  * @param randBasis              breeze RandBasis to use for generating breeze random numbers
  */
case class MultiTaskBagger(
                            method: MultiTaskLearner,
                            numBags: Int = -1,
                            useJackknife: Boolean = true,
                            biasLearner: Option[Learner] = None,
                            uncertaintyCalibration: Boolean = false,
                            randBasis: RandBasis = Rand
                          ) extends MultiTaskLearner {

  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): MultiTaskTrainingResult = {
    /* Make sure the training data are the same size */
    assert(inputs.forall(inputs.head.size == _.size))
    assert(inputs.size >= Bagger.minimumTrainingSize, s"We need to have at least ${Bagger.minimumTrainingSize} rows, only ${inputs.size} given")

    // if numBags is non-positive, set # bags = # inputs
    val actualBags = if (numBags > 0) numBags else inputs.size

    // Compute the number of instances of each training row in each training sample
    val dist = new Poisson(1.0)(randBasis)
    val Nib: Vector[Vector[Int]] = Vector.tabulate(actualBags) { _ =>
      Vector.tabulate(inputs.size) { _ =>
        dist.draw()
      }
    }
    val weightsActual = weights.getOrElse(Seq.fill(inputs.size)(1.0))

    val parIterator = new ParRange(Nib.indices)
    parIterator.tasksupport = new ExecutionContextTaskSupport(InterruptibleExecutionContext())
    val (models: ParSeq[MultiTaskModel], importances: ParSeq[Option[Vector[Double]]]) = parIterator.map { i =>
      val meta = method.train(inputs.toVector, labels, Some(Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)))
      (meta.getModel(), meta.getFeatureImportance())
    }.unzip

    val averageImportance: Option[Vector[Double]] = importances
      .reduce(combineImportance)
      .map(_.map(_ / importances.size))
    val trainingData = inputs.zip(labels.transpose)

    // Get bias model and rescale ratio for each label
    val (biasModels, ratios) = Seq.tabulate(labels.length) { i =>
      val thisLabelModels: ParSeq[Model[PredictionResult[Any]]] = models.map(_.getModels(i))
      val isRegression = models.head.getRealLabels(i)
      val helper = BaggerHelper(thisLabelModels, inputs.zip(labels(i)), Nib, useJackknife, uncertaintyCalibration)
      val biasModel = if (biasLearner.isDefined && isRegression) {
        Async.canStop()
        Some(biasLearner.get.train(helper.biasTraining).getModel().asInstanceOf[Model[PredictionResult[Double]]])
      } else None
      (biasModel, helper.ratio)
    }.unzip

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
      case _ => None
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

  lazy val model = new MultiTaskBaggedModel(models, Nib, useJackknife, biasModels)

  override def getFeatureImportance(): Option[Vector[Double]] = featureImportance

  override def getModel(): MultiTaskModel = model

  override def getModels(): Seq[Model[PredictionResult[Any]]] = {
    val realLabels: Seq[Boolean] = models.head.getRealLabels
    models.map(_.getModels).toVector.transpose.zipWithIndex.map { case (models: Vector[Model[PredictionResult[Any]]], i: Int) =>
      if (realLabels(i)) {
        new BaggedModel[Double](models.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], Nib, useJackknife, biasModels(i), rescaleRatios(i))
      } else {
        new BaggedModel[Any](models.par, Nib, useJackknife, biasModels(i), rescaleRatios(i))
      }
    }
  }

  // TODO: use trainingData and model to get predicted vs. actual, which can be used to get loss (see BaggedTrainingResult)
}

/**
  * Container holding a parallel sequence of models, each of which predicts on multiple labels.
  *
  * @param models       sequence of multi-models, one for each bag
  * @param Nib          matrix representing number of times each training datum appears in each bag
  * @param useJackknife whether to enable jackknife uncertainty estimate
  * @param biasModels   sequence of optional bias-correction models, one for each label
  */
class MultiTaskBaggedModel(
                            models: ParSeq[MultiTaskModel],
                            Nib: Vector[Vector[Int]],
                            useJackknife: Boolean,
                            biasModels: Seq[Option[Model[PredictionResult[Double]]]]
                          ) extends MultiTaskModel {

  lazy val groupedModels: Seq[BaggedModel[Any]] = Seq.tabulate(numLabels) { i =>
    val thisLabelsModels = models.map(_.getModels(i))
    if (getRealLabels(i)) {
      new BaggedModel[Double](thisLabelsModels.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], Nib, useJackknife, biasModels(i))
    } else {
      new BaggedModel(thisLabelsModels, Nib, useJackknife, biasModels(i))
    }
  }

  override def transform(inputs: Seq[Vector[Any]]) = MultiTaskBaggedResult(groupedModels.map(_.transform(inputs)), getRealLabels)

  override val numLabels: Int = models.head.numLabels

  override def getRealLabels: Seq[Boolean] = models.head.getRealLabels

  override def getModels: Seq[Model[PredictionResult[Any]]] = groupedModels
}
