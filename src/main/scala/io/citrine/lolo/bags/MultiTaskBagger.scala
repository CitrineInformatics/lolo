package io.citrine.lolo.bags

import breeze.stats.distributions.{Poisson, Rand, RandBasis}
import io.citrine.lolo._
import io.citrine.lolo.trees.multitask.MultiModel
import io.citrine.lolo.util.{Async, InterruptibleExecutionContext}

import scala.collection.parallel.ExecutionContextTaskSupport
import scala.collection.parallel.immutable.{ParRange, ParSeq}

/**
  * Create an ensemble of multi-task models
  *
  * @param method  learner to train each model in the ensemble
  * @param numBags number of models in the ensemble
  */
case class MultiTaskBagger(
                            method: MultiTaskLearner,
                            numBags: Int = -1,
                            useJackknife: Boolean = true,
                            biasLearner: Option[Learner] = None,
                            uncertaintyCalibration: Boolean = false,
                            randBasis: RandBasis = Rand
                          ) extends MultiTaskLearner {

  override val singleModel: Boolean = method.singleModel

  private def combineImportance(v1: Option[Vector[Double]], v2: Option[Vector[Double]]): Option[Vector[Double]] = {
    (v1, v2) match {
      case (None, None) => None
      case (Some(v1: Vector[Double]), Some(v2: Vector[Double])) => Some(v1.zip(v2).map(p => p._1 + p._2))
      case _ => None
    }
  }

  /**
    * Draw with replacement from the training data for each model
    *
    * @param inputs  to train on
    * @param weights for the training rows, if applicable
    * @return a model
    */
  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[AbstractBaggedTrainingResult] = {
    /* Make sure the training data is the same size */
    assert(inputs.forall(inputs.head.size == _.size))
    assert(inputs.size >= Bagger.minimumTrainingSize, s"We need to have at least ${Bagger.minimumTrainingSize} rows, only ${inputs.size} given")

    /* Use unit weights if none are specified */
    val weightsActual = weights.getOrElse(Seq.fill(inputs.size)(1.0))

    /* Set default number of bags */
    val actualBags = if (numBags > 0) {
      numBags
    } else {
      inputs.size
    }

    /* Compute the number of instances of each training row in each training sample */
    val dist = new Poisson(1.0)(randBasis)
    val Nib = Vector.tabulate(actualBags) { _ =>
      Vector.tabulate(inputs.size) { _ =>
        dist.draw()
      }
    }

    /* Learn the actual models in parallel */
    val parIterator = new ParRange(0 until actualBags)
    parIterator.tasksupport = new ExecutionContextTaskSupport(InterruptibleExecutionContext())
    val (models: ParSeq[Seq[Model[PredictionResult[Any]]]], importances: ParSeq[Seq[Option[Vector[Double]]]]) = parIterator.map { i =>
      // Train the model
      val meta = method.train(inputs.toVector, labels, Some(Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)))
      assert(!singleModel || meta.length == 1)
      // Extract the model and feature importance from the TrainingResult
      (meta.map(_.getModel()), meta.map(_.getFeatureImportance()))
    }.unzip

    if (singleModel) {
      // TODO: the singleModel logic here is similar to the multi-model logic, below. Try to consolidate.
      val flatModels = models.map(_.head.asInstanceOf[MultiModel])
      val flatImportances = importances.map(_.head)
      val averageImportance: Option[Vector[Double]] = flatImportances.reduce {
        combineImportance
      }.map(_.map(_ / importances.size))
      val trainingData = inputs.zip(labels)

      val (biasModels, ratios) = Seq.tabulate(labels.length) { i =>
        val thisLabelModels: ParSeq[Model[PredictionResult[Any]]] = flatModels.map(_.getModels(i))
        val isRegression = flatModels.head.getRealLabels(i)
        val helper = BaggerHelper(thisLabelModels, inputs.zip(labels(i)), Nib, useJackknife, uncertaintyCalibration)
        val biasModel = if (biasLearner.isDefined && isRegression) {
          Some(biasLearner.get.train(helper.biasTraining).getModel().asInstanceOf[Model[PredictionResult[Double]]])
        } else None
        (biasModel, helper.ratio)
      }.unzip

      return Seq(
        new MultiTaskBaggedTrainingResult(
          models = flatModels,
          featureImportance = averageImportance,
          Nib = Nib,
          trainingData = trainingData,
          useJackknife = useJackknife,
          biasModels = biasModels,
          rescaleRatios = ratios
        )
      )
    }

    /* Wrap the models in a BaggedModel object
     *
     * Transpose the models and importances so the bags are the inner index and the labels are the outer index.
     * For each label, emit a BaggedTrainingResult
     */
    type arg = ((ParSeq[Model[PredictionResult[Any]]], Seq[Option[Vector[Double]]]), Int)
    models.transpose.zip(importances.seq.transpose).zipWithIndex.map { xx: arg =>
      xx match {
        case ((m: ParSeq[Model[PredictionResult[Any]]], i: Seq[Option[Vector[Double]]]), k: Int) =>
          val averageImportance: Option[Vector[Double]] = i.reduce {
            combineImportance
          }.map(_.map(_ / importances.size))
          val trainingData = inputs.zip(labels(k))
          val helper = BaggerHelper(m, trainingData, Nib, useJackknife, uncertaintyCalibration)

          Async.canStop()
          if (!helper.isRegression) {
            new BaggedTrainingResult[Any](m, averageImportance, Nib, trainingData, useJackknife)
          } else {
            if (biasLearner.isEmpty) {
              new BaggedTrainingResult[Double](m.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], averageImportance, Nib, trainingData, useJackknife, None, helper.ratio)
            } else {
              Async.canStop()
              val biasModel = biasLearner.get.train(helper.biasTraining).getModel().asInstanceOf[Model[PredictionResult[Double]]]
              Async.canStop()
              new BaggedTrainingResult[Double](m.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], averageImportance, Nib, trainingData, useJackknife, Some(biasModel), helper.ratio)
            }
          }
      }
      }
      .seq
  }
}

class MultiTaskBaggedTrainingResult(
                                   models: ParSeq[MultiModel],
                                   featureImportance: Option[Vector[Double]],
                                   Nib: Vector[Vector[Int]],
                                   trainingData: Seq[(Vector[Any], Seq[Any])],
                                   useJackknife: Boolean,
                                   biasModels: Seq[Option[Model[PredictionResult[Double]]]],
                                   rescaleRatios: Seq[Double]
                                   ) extends AbstractBaggedTrainingResult {

  lazy val model = new MultiTaskBaggedModel(models, Nib, useJackknife, biasModels)

  override def getFeatureImportance(): Option[Vector[Double]] = featureImportance

  override def getModel(): Model[PredictionResult[Any]] = model

  // TODO: use trainingData and model to get predicted vs. actual, although the abstraction might not allow
  //  for sequence-valued predicted and actual values.

}

class MultiTaskBaggedModel(
                          models: ParSeq[MultiModel],
                          Nib: Vector[Vector[Int]],
                          useJackknife: Boolean,
                          biasModels: Seq[Option[Model[PredictionResult[Double]]]]
                          ) extends MultiModel {

  lazy val groupedModels: Seq[BaggedModel[Any]] = Seq.tabulate(numLabels) { i =>
    val thisLabelsModels = models.map(_.getModels(i))
    if (getRealLabels(i)) {
      new BaggedModel[Double](thisLabelsModels.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], Nib, useJackknife, biasModels(i))
    } else {
      new BaggedModel(thisLabelsModels, Nib, useJackknife, biasModels(i))
    }
  }

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult[Seq[Any]] = MultiTaskBaggedResult(groupedModels.map(_.transform(inputs)), Nib.length)

  override val numLabels: Int = models.head.numLabels

  override def getRealLabels: Seq[Boolean] = models.head.getRealLabels

  override def getModels: Seq[Model[PredictionResult[Any]]] = groupedModels

}