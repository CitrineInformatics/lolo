package io.citrine.lolo.bags

import breeze.stats.distributions.{Poisson, Rand, RandBasis}
import io.citrine.lolo._
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
  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[BaggedTrainingResult[Any]] = {
    /* Make sure the training data is the same size */
    assert(inputs.forall(inputs.head.size == _.size))
    assert(inputs.size > 8, s"We need to have at least 8 rows, only ${inputs.size} given")

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
      // Extract the model and feature importance from the TrainingResult
      (meta.map(_.getModel()), meta.map(_.getFeatureImportance()))
    }.unzip

    /* Wrap the models in a BaggedModel object
     *
     * Transpose the models and importances so the bags are the inner index and the labels are the outer index.
     * Foreach label, emit a BaggedTrainingResult
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
            new BaggedTrainingResult[Any](m, averageImportance, Nib, inputs.zip(labels(k)), useJackknife)
          } else {
            if (biasLearner.isEmpty) {
              new BaggedTrainingResult[Double](m.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], averageImportance, Nib, trainingData, useJackknife, None, helper.ratio)
            } else {
              Async.canStop()
              val baggedModel = new BaggedModel[Double](m.asInstanceOf[ParSeq[Model[PredictionResult[Double]]]], Nib, useJackknife, None, helper.ratio)
              Async.canStop()
              val baggedRes = baggedModel.transform(trainingData.map(_._1))
              Async.canStop()
              val foo = baggedRes.getUncertainty()
              val biasTraining = trainingData.zip(
                baggedRes.getExpected().zip(baggedRes.getUncertainty().get)
              ).flatMap { case ((f, a), (p, u)) =>
                if (a == null || (a.isInstanceOf[Double] && a.asInstanceOf[Double].isNaN)) {
                  None
                } else {
                  // Math.E is only statistically correct.  It should be actualBags / Nib.transpose(i).count(_ == 0)
                  // Or, better yet, filter the bags that don't include the training example
                  val bias = Math.E * Math.max(Math.abs(p.asInstanceOf[Double] - a.asInstanceOf[Double]) - u.asInstanceOf[Double], 0.0)
                  Some((f, bias))
                }
              }
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
