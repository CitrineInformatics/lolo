package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo.{Learner, Model, PredictionResult, withScores, withUncertainty}

import scala.collection.parallel.immutable.ParSeq

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data
  *
  * Created by maxhutch on 11/14/16.
  * @param method learner to train each model in the ensemble
  * @param numBags number of models in the ensemble
  */
class Bagger(method: Learner, var numBags: Int = -1) extends Learner {

  /**
    * Draw with replacement from the training data for each model
    *
    * @param trainingData to train on
    * @param weights for the training rows, if applicable
    * @return a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): BaggedModel = {
    /* Make sure the training data is the same size */
    assert(trainingData.forall(trainingData.head._1.size == _._1.size))

    /* Use unit weights if none are specified */
    val weightsActual = weights.getOrElse(Seq.fill(trainingData.size)(1.0))

    /* Set default number of bags */
    val actualBags = if (numBags > 0) {
      numBags
    } else {
      trainingData.size
    }

    /* Compute the number of instances of each training row in each training sample */
    val dist = new Poisson(1.0)
    val Nib = Vector.tabulate(actualBags) { i =>
      Vector.tabulate(trainingData.size) { j =>
        dist.draw()
      }
    }

    /* Learn the actual models */
    val models = (0 until actualBags).par.map { i =>
      method.train(trainingData, Some(Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)))
    }

    /* Wrap the models in a BaggedModel object */
    new BaggedModel(models, Nib)
  }
}

/**
  * Container holding a parallel sequence of models and the sample counts used to train them
  * @param models in this bagged model
  * @param Nib training sample counts
  */
class BaggedModel(models: ParSeq[Model], Nib: Vector[Vector[Int]]) extends Model {

  /**
    * Apply each model to the outputs and wrap them up
    * @param inputs to apply the model to
    * @return a predictionresult that includes uncertainties and scores
    */
  override def transform(inputs: Seq[Vector[Any]]): BaggedResult = {
    assert(inputs.forall(_.size == inputs.head.size))
    println(s"Applying model on ${inputs.head.size} inputs")
    new BaggedResult(models.map(model => model.transform(inputs)).seq, Nib)
  }

  /**
    * Average the importances across the ensemble of models
    * @return feature importances as an array of doubles
    */
  override def getFeatureImportance(): Array[Double] = {
    val importances: Array[Double] = models.map(model => model.getFeatureImportance()).reduce { (v1, v2) =>
      v1.zip(v2).map(p => p._1 + p._2)
    }
    importances.map(_ / models.size)
  }
}

/**
  * Container with model-wise predictions and logic to compute variances and training row scores
  * @param predictions for each constituent model
  * @param NibIn the sample matrix as (N_models x N_training)
  */
class BaggedResult(predictions: Seq[PredictionResult], NibIn: Vector[Vector[Int]]) extends PredictionResult
  with withUncertainty with withScores {

  /**
    * Return the ensemble average or maximum vote
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = expected

  /**
    * Return jackknife-based variance estimates
    * @return uncertainty of each prediction
    */
  override def getUncertainty(): Seq[Any] = uncertainty

  /**
    * Return IJ scores
    * @return training row scores of each prediction
    */
  override def getScores(): Seq[Seq[Double]] = scores

  /* Subtract off 1 to make correlations easier; transpose to be prediction-wise */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  /* Make a matrix of the tree-wise predictions */
  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose

  /* For checking the type of the prediction */
  lazy val rep: Any = expectedMatrix.head.head

  /* Extract the prediction by averaging for regression, taking the most popular response for classification */
  lazy val expected = rep match {
    case x: Double => expectedMatrix.map(ps => ps.asInstanceOf[Seq[Double]].sum / ps.size)
    case x: Any => expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size).maxBy(_._2)._1).seq
  }

  /* Compute the uncertainties one prediction at a time */
  lazy val uncertainty = rep match {
    case x: Double => expectedMatrix.zip(expected).map { case (treePredictions, meanPrediction) =>
      variance(meanPrediction.asInstanceOf[Double], treePredictions.asInstanceOf[Seq[Double]], Nib)
    }
    case x: Any => Seq.fill(expected.size)(1.0)
  }

  /* Compute the scores one prediction at a time */
  lazy val scores: Seq[Vector[Double]] = rep match {
    case x: Double => expectedMatrix.zip(expected).map { case (treePredictions, meanPrediction) =>
      scores(meanPrediction.asInstanceOf[Double], treePredictions.asInstanceOf[Seq[Double]], Nib)
    }
    case x: Any => Seq.fill(expected.size)(Vector.fill(Nib.size)(0.0))
  }

  /**
    * Compute the variance of a prediction as the average of bias corrected IJ and J variance estimates
    * @param meanPrediction over the models
    * @param modelPredictions prediction of each model
    * @param Nib sampling matrix
    * @return the estimated variance
    */
  def variance(meanPrediction: Double, modelPredictions: Seq[Double], Nib: Seq[Vector[Int]]): Double = {
    val diff = modelPredictions.map(_ - meanPrediction)

    /* Compute the infintesimal jackknife variance estimate */
    val varianceIJ: Double = Nib.map { v =>
      val cov = v.zip(diff).map(p2 => p2._1 * p2._2).sum / modelPredictions.size
      cov * cov
    }.sum

    /* Compute the jackknife-after-bootstrap variance estimate */
    val varianceJ = (Nib.size - 1.0) / Nib.size * Nib.map { v =>
      val predictionsWithoutV: Seq[Double] = v.zip(modelPredictions).filter(_._1 == -1.0).map(_._2)
      Math.pow(predictionsWithoutV.sum / predictionsWithoutV.size - meanPrediction, 2)
    }.sum

    /* Compute the first order bias correction for the variance estimators */
    val correction = diff.map(Math.pow(_, 2)).sum * Nib.size / (modelPredictions.size * modelPredictions.size)

    /* Mix the IJ and J estimators with their bias corrections */
    val result = (varianceIJ + varianceJ - Math.E * correction) / 2.0
    if (result < 0) {
      println("Warning: negative variance; increase the number of trees")
      0.0
    } else {
      result
    }
  }

  /**
    * Compute the IJ training row scores for a prediction
    *
    * @param meanPrediction across the models
    * @param modelPredictions predictions of each model
    * @param Nib sample matrix
    * @return the score of each training row as a vector of doubles
    */
  def scores(meanPrediction: Double, modelPredictions: Seq[Double], Nib: Seq[Vector[Int]]): Vector[Double] = {
    val diff = modelPredictions.map(_ - meanPrediction)

    /* Compute the IJ score for each row */
    Nib.map { v =>
      Math.abs(v.zip(diff).map(p2 => p2._1 * p2._2).sum / modelPredictions.size)
    }.toVector
  }
}