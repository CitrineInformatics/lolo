package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo._
import io.citrine.lolo.util.{Async, InterruptibleExecutionContext}

import scala.collection.parallel.ExecutionContextTaskSupport
import scala.collection.parallel.immutable.ParSeq

/**
  * Create an ensemble of multi-task models
  *
  * @param method  learner to train each model in the ensemble
  * @param numBags number of models in the ensemble
  */
class MultiTaskBagger(
                       method: MultiTaskLearner,
                       numBags: Int = -1,
                       useJackknife: Boolean = true,
                       biasLearner: Option[Learner] = None
                     ) extends MultiTaskLearner {

  setHypers(Map("useJackknife" -> useJackknife, "numBags" -> numBags))

  override def setHypers(moreHypers: Map[String, Any]): this.type = {
    method.setHypers(moreHypers)
    super.setHypers(moreHypers)
  }

  override def getHypers(): Map[String, Any] = {
    method.getHypers() ++ hypers
  }

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
  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]] = None): Seq[BaggedTrainingResult] = {
    /* Make sure the training data is the same size */
    assert(inputs.forall(inputs.head.size == _.size))
    assert(inputs.size > 8, s"We need to have at least 8 rows, only ${inputs.size} given")

    /* Use unit weights if none are specified */
    val weightsActual = weights.getOrElse(Seq.fill(inputs.size)(1.0))

    /* Set default number of bags */
    val actualBags = if (hypers("numBags").asInstanceOf[Int] > 0) {
      hypers("numBags").asInstanceOf[Int]
    } else {
      inputs.size
    }

    /* Compute the number of instances of each training row in each training sample */
    val dist = new Poisson(1.0)
    val Nib = Vector.tabulate(actualBags) { _ =>
      Vector.tabulate(inputs.size) { _ =>
        dist.draw()
      }
    }

    /* Learn the actual models in parallel */
    val parIterator = (0 until actualBags).par
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
    models.transpose.zip(importances.seq.transpose).zipWithIndex.map { case ((m, i: Seq[Option[Vector[Double]]]), k) =>
      val averageImportance: Option[Vector[Double]] = i.reduce {
        combineImportance
      }.map(_.map(_ / importances.size))
      val trainingData = inputs.zip(labels(k))
      Async.canStop()
      if (biasLearner.isEmpty || !labels(k).head.isInstanceOf[Double]) {
        new BaggedTrainingResult(m, getHypers(), averageImportance, Nib, inputs.zip(labels(k)), hypers("useJackknife").asInstanceOf[Boolean])
      } else {
        Async.canStop()
        val baggedModel = new BaggedModel(m, Nib, hypers("useJackknife").asInstanceOf[Boolean])
        Async.canStop()
        val baggedRes = baggedModel.transform(trainingData.map(_._1))
        Async.canStop()
        val biasTraining = trainingData.zip(
          baggedRes.getExpected().zip(baggedRes.getUncertainty().get)
        ).map { case ((f, a), (p, u)) =>
          // Math.E is only statistically correct.  It should be actualBags / Nib.transpose(i).count(_ == 0)
          // Or, better yet, filter the bags that don't include the training example
          val bias = Math.E * Math.max(Math.abs(p.asInstanceOf[Double] - a.asInstanceOf[Double]) - u.asInstanceOf[Double], 0.0)
          (f, bias)
        }
        Async.canStop()
        val biasModel = biasLearner.get.train(biasTraining).getModel()
        Async.canStop()

        new BaggedTrainingResult(m, getHypers(), averageImportance, Nib, trainingData, hypers("useJackknife").asInstanceOf[Boolean], Some(biasModel))
      }
    }.seq
  }
}
