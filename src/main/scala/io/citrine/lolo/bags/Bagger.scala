package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo.{Learner, Model, PredictionResult, withScores, withUncertainty}

import scala.collection.parallel.immutable.ParSeq

/**
  * Created by maxhutch on 11/14/16.
  */
class Bagger(method: Learner, var numBags: Int = -1) extends Learner {
  override def train(trainingDataIn: Seq[(Vector[Any], Any)], weightsIn: Option[Seq[Double]] = None): BaggedModel = {

    val weightsRaw = weightsIn.getOrElse(Seq.fill(trainingDataIn.size)(1.0))
    // val (trainingData, weights) = trainingDataIn.zip(weightsRaw).groupBy(_._1).mapValues(_.map(_._2).sum).unzip
    val trainingData = trainingDataIn
    val weights = weightsRaw
    // println(weights)

    val actualBags = if (numBags > 0) {
      numBags
    } else {
      trainingData.size
    }

    assert(trainingData.forall(trainingData.head._1.size == _._1.size))
    println(s"Trained a model with ${trainingData.head._1.size} inputs")


    val dist = new Poisson(1.0)
    val Nib = Vector.tabulate(actualBags) { i =>
      Vector.tabulate(trainingData.size) { j =>
        dist.draw()
      }
    }

    val models = (0 until actualBags).par.map { i =>
      method.train(trainingData.toSeq, Some(Nib(i).zip(weights).map(p => p._1.toDouble * p._2)))
    }

    new BaggedModel(models, Nib)
  }
}

class BaggedModel(models: ParSeq[Model], Nib: Vector[Vector[Int]]) extends Model {

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult = {
    assert(inputs.forall(_.size == inputs.head.size))
    println(s"Applying model on ${inputs.head.size} inputs")
    new BaggedResult(models.map(model => model.transform(inputs)).seq, Nib)
  }

  override def getFeatureImportance(): Array[Double] = {
    val importances: Array[Double] = models.map(model => model.getFeatureImportance()).reduce { (v1, v2) =>
      v1.zip(v2).map(p => p._1 + p._2)
    }
    importances.map(_ / models.size)
  }
}

class BaggedResult(predictions: Seq[PredictionResult], NibIn: Vector[Vector[Int]]) extends PredictionResult
  with withUncertainty with withScores {

  override def getExpected(): Seq[Any] = expected

  override def getUncertainty(): Seq[Any] = uncertainty

  override def getScores(): Seq[Seq[Double]] = scores

  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose

  lazy val rep: Any = expectedMatrix.head.head

  lazy val expected = rep match {
    case x: Double => expectedMatrix.map(ps => ps.asInstanceOf[Seq[Double]].sum / ps.size)
    case x: Any => expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size).maxBy(_._2)._1).seq
  }

  lazy val uncertainty = rep match {
    case x: Double => expectedMatrix.zip(expected).map { case (treePredictions, meanPrediction) =>
      variance(meanPrediction.asInstanceOf[Double], treePredictions.asInstanceOf[Seq[Double]], Nib)
    }
    case x: Any => Seq.fill(expected.size)(1.0)
  }

  lazy val scores: Seq[Vector[Double]] = rep match {
    case x: Double => expectedMatrix.zip(expected).map { case (treePredictions, meanPrediction) =>
      scores(meanPrediction.asInstanceOf[Double], treePredictions.asInstanceOf[Seq[Double]], Nib)
    }
    case x: Any => Seq.fill(expected.size)(Vector.fill(Nib.size)(0.0))
  }

  def variance(meanPrediction: Double, treePredictions: Seq[Double], Nib: Seq[Vector[Int]]): Double = {
    val diff = treePredictions.map(_ - meanPrediction)

    /* Compute the infintesimal jackknife variance estimate */
    val varianceIJ: Double = Nib.map { v =>
      val cov = v.zip(diff).map(p2 => p2._1 * p2._2).sum / treePredictions.size
      cov * cov
    }.sum

    /* Compute the jackknife-after-bootstrap variance estimate */
    val varianceJ = (Nib.size - 1.0) / Nib.size * Nib.map { v =>
      val predictionsWithoutV: Seq[Double] = v.zip(treePredictions).filter(_._1 == -1.0).map(_._2)
      Math.pow(predictionsWithoutV.sum / predictionsWithoutV.size - meanPrediction, 2)
    }.sum

    /* Compute the first order bias correction for the variance estimators */
    val correction = diff.map(Math.pow(_, 2)).sum * Nib.size / (treePredictions.size * treePredictions.size)
    /* Mix the IJ and J estimators with their bias corrections */
    val result = (varianceIJ + varianceJ - Math.E * correction) / 2.0
    if (result < 0) {
      println("Warning: negative variance; increase the number of trees")
      0.0
    } else {
      result
    }
  }

  def scores(meanPrediction: Double, treePredictions: Seq[Double], Nib: Seq[Vector[Int]]): Vector[Double] = {
    val diff = treePredictions.map(_ - meanPrediction)

    /* Compute the IJ score for each row */
    Nib.map { v =>
      Math.abs(v.zip(diff).map(p2 => p2._1 * p2._2).sum / treePredictions.size)
    }.toVector
  }
}