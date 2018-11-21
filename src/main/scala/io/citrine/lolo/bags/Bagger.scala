package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix, DenseVector, min, norm}
import breeze.numerics.abs
import breeze.stats.distributions.Poisson
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.util.{Async, InterruptibleExecutionContext}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

import scala.collection.parallel.ExecutionContextTaskSupport
import scala.collection.parallel.immutable.ParSeq

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data
  *
  * Created by maxhutch on 11/14/16.
  *
  * @param method  learner to train each model in the ensemble
  * @param numBags number of models in the ensemble
  */
case class Bagger(
              method: Learner,
              numBags: Int = -1,
              useJackknife: Boolean = true,
              biasLearner: Option[Learner] = None
            ) extends Learner {

  override def getHypers(): Map[String, Any] = {
    method.getHypers() ++ Map("useJackknife" -> useJackknife, "numBags" -> numBags)
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
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): BaggedTrainingResult = {
    /* Make sure the training data is the same size */
    assert(trainingData.forall(trainingData.head._1.size == _._1.size))
    assert(trainingData.size > 8, s"We need to have at least 8 rows, only ${trainingData.size} given")

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

    /* Learn the actual models in parallel */
    val parIterator = (0 until actualBags).par
    parIterator.tasksupport = new ExecutionContextTaskSupport(InterruptibleExecutionContext())
    val (models, importances: ParSeq[Option[Vector[Double]]]) = parIterator.map { i =>
      // Train the model
      val meta = method.train(trainingData.toVector, Some(Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)))
      // Extract the model and feature importance from the TrainingResult
      (meta.getModel(), meta.getFeatureImportance())
    }.unzip

    // Average the feature importances
    val averageImportance: Option[Vector[Double]] = importances.reduce{combineImportance}.map(_.map(_ / importances.size))

    /* Wrap the models in a BaggedModel object */
    if (biasLearner.isEmpty) {
      Async.canStop()
      new BaggedTrainingResult(models, getHypers(), averageImportance, Nib, trainingData, useJackknife)
    } else {
      Async.canStop()
      val baggedModel = new BaggedModel(models, Nib, useJackknife)
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

      new BaggedTrainingResult(models, getHypers(), averageImportance, Nib, trainingData, useJackknife, Some(biasModel))
    }
  }
}

@SerialVersionUID(999L)
class BaggedTrainingResult(
                            models: ParSeq[Model[PredictionResult[Any]]],
                            hypers: Map[String, Any],
                            featureImportance: Option[Vector[Double]],
                            Nib: Vector[Vector[Int]],
                            trainingData: Seq[(Vector[Any], Any)],
                            useJackknife: Boolean,
                            biasModel: Option[Model[PredictionResult[Any]]] = None
                          )
  extends TrainingResult {
  lazy val NibT = Nib.transpose
  lazy val model = new BaggedModel(models, Nib, useJackknife, biasModel)
  lazy val rep = trainingData.find(_._2 != null).get._2
  lazy val predictedVsActual = trainingData.zip(NibT).flatMap { case ((f, l), nb) =>
    val oob = models.zip(nb).filter(_._2 == 0)
    if (oob.isEmpty || l == null || (l.isInstanceOf[Double] && l.asInstanceOf[Double].isNaN) ) {
      Seq()
    } else {
      val predicted = l match {
        case _: Double => oob.map(_._1.transform(Seq(f)).getExpected().head.asInstanceOf[Double]).sum / oob.size
        case _: Any => oob.map(_._1.transform(Seq(f)).getExpected().head).groupBy(identity).maxBy(_._2.size)._1
      }
      Seq((f, predicted, l))
    }
  }

  lazy val loss: Double = rep match {
    case x: Double => Math.sqrt(predictedVsActual.map(d => Math.pow(d._2.asInstanceOf[Double] - d._3.asInstanceOf[Double], 2)).sum / predictedVsActual.size)
    case x: Any =>
      val f1 = ClassificationMetrics.f1scores(predictedVsActual)
      if (f1 > 0.0) 1.0 / f1 - 1.0 else Double.MaxValue
  }

  /**
    * Average the influences across the ensemble of models
    *
    * @return feature influences as an array of doubles
    */
  override def getFeatureImportance(): Option[Vector[Double]] = featureImportance

  override def getModel(): BaggedModel = model

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Any, Any)]] = Some(predictedVsActual)

  override def getLoss(): Option[Double] = Some(loss)

  /**
    * Get the hyperparameters used to train this model
    *
    * @return hypers set for model
    */
  override def getHypers(): Map[String, Any] = hypers
}

/**
  * Container holding a parallel sequence of models and the sample counts used to train them
  *
  * @param models in this bagged model
  * @param Nib    training sample counts
  */
@SerialVersionUID(1000L)
class BaggedModel(
                   models: ParSeq[Model[PredictionResult[Any]]],
                   Nib: Vector[Vector[Int]],
                   useJackknife: Boolean,
                   biasModel: Option[Model[PredictionResult[Any]]] = None
                 ) extends Model[BaggedResult] {

  /**
    * Apply each model to the outputs and wrap them up
    *
    * @param inputs to apply the model to
    * @return a predictionresult that includes uncertainties and scores
    */
  override def transform(inputs: Seq[Vector[Any]]): BaggedResult = {
    assert(inputs.forall(_.size == inputs.head.size))
    // println(s"Applying model on ${inputs.size} inputs of length ${inputs.head.size}")
    val bias = if (biasModel.isDefined) {
      Some(biasModel.get.transform(inputs).getExpected().asInstanceOf[Seq[Double]])
    } else {
      None
    }
    new BaggedResult(models.map(model => model.transform(inputs)).seq, Nib, useJackknife, bias, inputs.head)
  }
}


/**
  * Container with model-wise predictions and logic to compute variances and training row scores
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  * @param repInput    representative input
  */
class BaggedResult(
                    predictions: Seq[PredictionResult[Any]],
                    NibIn: Vector[Vector[Int]],
                    useJackknife: Boolean,
                    bias: Option[Seq[Double]] = None,
                    repInput: Vector[Any]
                  ) extends PredictionResult[Any] {

  /**
    * Return the ensemble average or maximum vote
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = expected

  /**
    * Return jackknife-based variance estimates
    *
    * @return uncertainty of each prediction
    */
  override def getUncertainty(): Option[Seq[Any]] = Some(uncertainty)

  /**
    * Return IJ scores
    *
    * @return training row scores of each prediction
    */
  override def getInfluenceScores(actuals: Seq[Any]): Option[Seq[Seq[Double]]] = {
    rep match {
      case x: Double =>
        Some(influences(
          expected.asInstanceOf[Seq[Double]].toVector,
          actuals.toVector.asInstanceOf[Vector[Double]],
          expectedMatrix.asInstanceOf[Seq[Seq[Double]]],
          NibJMat,
          NibIJMat
        ))
      case x: Any => None
    }
  }

  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(scores)

  /* Subtract off 1 to make correlations easier; transpose to be prediction-wise */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  /* Make a matrix of the tree-wise predictions */
  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose

  /* For checking the type of the prediction */
  lazy val rep: Any = expectedMatrix.head.head

  /* Extract the prediction by averaging for regression, taking the most popular response for classification */
  lazy val expected = rep match {
    case x: Double => expectedMatrix.map(ps => ps.asInstanceOf[Seq[Double]].sum / ps.size)
    case x: Any => expectedMatrix.map(ps => ps.groupBy(identity).maxBy(_._2.size)._1).seq
  }

  /* This matrix is used to compute the jackknife variance */
  lazy val NibJMat = new DenseMatrix[Double](Nib.head.size, Nib.size,
    Nib.flatMap { v =>
      val itot = 1.0 / v.size
      val icount = 1.0 / v.count(_ == -1.0)
      v.map(n => if (n == -1) icount - itot else -itot)
    }.toArray
  )

  /* This matrix is used to compute the IJ variance */
  lazy val NibIJMat = new DenseMatrix[Double](Nib.head.size, Nib.size,
    Nib.flatMap { v =>
      val itot = 1.0 / v.size
      val vtot = v.sum.toDouble / (v.size * v.size)
      v.map(n => (n * itot - vtot))
    }.toArray
  )

  /* Compute the uncertainties one prediction at a time */
  lazy val uncertainty = rep match {
    case x: Double =>
      val sigma2: Seq[Double] = if (useJackknife) {
        variance(expected.asInstanceOf[Seq[Double]].toVector, expectedMatrix.asInstanceOf[Seq[Seq[Double]]], NibJMat, NibIJMat)
      } else {
        Seq.fill(expected.size)(0.0)
      }
      sigma2.zip(bias.getOrElse(Seq.fill(expected.size)(0.0))).map(p => Math.sqrt(p._2 * p._2 + p._1))
    case x: Any =>
      expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size.toDouble / ps.size))
  }

  /* Compute the scores one prediction at a time */
  lazy val scores: Seq[Vector[Double]] = rep match {
    case x: Double =>
      scores(expected.asInstanceOf[Seq[Double]].toVector, expectedMatrix.asInstanceOf[Seq[Seq[Double]]], NibJMat, NibIJMat).map(_.map(Math.sqrt))
    case x: Any => Seq.fill(expected.size)(Vector.fill(Nib.size)(0.0))
  }

  /**
    * Compute the variance of a prediction as the average of bias corrected IJ and J variance estimates
    *
    * @param meanPrediction   over the models
    * @param modelPredictions prediction of each model
    * @param NibJ             sampling matrix for the jackknife-after-bootstrap estimate
    * @param NibIJ            sampling matrix for the infinitesimal jackknife estimate
    * @return the estimated variance
    */
  def variance(
                meanPrediction: Vector[Double],
                modelPredictions: Seq[Seq[Double]],
                NibJ: DenseMatrix[Double],
                NibIJ: DenseMatrix[Double]
              ): Seq[Double] = {
    scores(meanPrediction, modelPredictions, NibJ, NibIJ).map(_.sum)
  }

  /**
    * Compute the IJ training row scores for a prediction
    *
    * @param meanPrediction   over the models
    * @param modelPredictions prediction of each model
    * @param NibJ             sampling matrix for the jackknife-after-bootstrap estimate
    * @param NibIJ            sampling matrix for the infinitesimal jackknife estimate
    * @return the score of each training row as a vector of doubles
    */
  def scores(
              meanPrediction: Vector[Double],
              modelPredictions: Seq[Seq[Double]],
              NibJ: DenseMatrix[Double],
              NibIJ: DenseMatrix[Double]
            ): Seq[Vector[Double]] = {
    /* Stick the predictions in a breeze matrix */
    val predMat = new DenseMatrix[Double](modelPredictions.head.size, modelPredictions.size, modelPredictions.flatten.toArray)

    /* These operations are pulled out of the loop and extra-verbose for performance */
    val JMat = NibJ.t * predMat
    Async.canStop()
    val JMat2 = JMat :* JMat * ((Nib.size - 1.0) / Nib.size)
    Async.canStop()
    val IJMat = NibIJ.t * predMat
    Async.canStop()
    val IJMat2 = IJMat :* IJMat
    Async.canStop()
    val arg = IJMat2 + JMat2
    Async.canStop()

    /* Avoid division in the loop */
    val inverseSize = 1.0 / modelPredictions.head.size

    (0 until modelPredictions.size).map { i =>
      Async.canStop()
      /* Compute the first order bias correction for the variance estimators */
      val correction = Math.pow(inverseSize * norm(predMat(::, i) - meanPrediction(i)), 2)

      /* The correction is prediction dependent, so we need to operate on vectors */
      val variancePerRow: DenseVector[Double] = 0.5 * (arg(::, i) - Math.E * correction)

      /* Impose a floor in case any of the variances are negative (hacked to work in breeze) */
      val floor: Double = Math.min(0, -min(variancePerRow))
      val rezero: DenseVector[Double] = variancePerRow - floor
      0.5 * (rezero + abs(rezero)) + floor
    }.map(_.toScalaVector())
  }

  /**
    * Compute the IJ training row scores for a prediction
    *
    * @param meanPrediction   over the models
    * @param modelPredictions prediction of each model
    * @param NibJ             sampling matrix for the jackknife-after-bootstrap estimate
    * @param NibIJ            sampling matrix for the infinitesimal jackknife estimate
    * @return the score of each training row as a vector of doubles
    */
  def influences(
                  meanPrediction: Vector[Double],
                  actualPrediction: Vector[Double],
                  modelPredictions: Seq[Seq[Double]],
                  NibJ: DenseMatrix[Double],
                  NibIJ: DenseMatrix[Double]
                ): Seq[Vector[Double]] = {
    /* Stick the predictions in a breeze matrix */
    val predMat = new DenseMatrix[Double](modelPredictions.head.size, modelPredictions.size, modelPredictions.flatten.toArray)

    /* These operations are pulled out of the loop and extra-verbose for performance */
    val JMat = NibJ.t * predMat
    val IJMat = NibIJ.t * predMat
    val arg = IJMat + JMat

    /* Avoid division in the loop */
    val inverseSize = 1.0 / modelPredictions.head.size

    (0 until modelPredictions.size).map { i =>
      /* Compute the first order bias correction for the variance estimators */
      val correction = 0.0 // inverseSize * norm(predMat(::, i) - meanPrediction(i))

      /* The correction is prediction dependent, so we need to operate on vectors */
      val influencePerRow: DenseVector[Double] = Math.signum(actualPrediction(i) - meanPrediction(i)) * 0.5 * (arg(::, i) - Math.E * correction)

      /* Impose a floor in case any of the variances are negative (hacked to work in breeze) */
      // val floor: Double = Math.min(0, -min(variancePerRow))
      // val rezero: DenseVector[Double] = variancePerRow - floor
      // 0.5 * (rezero + abs(rezero)) + floor
      influencePerRow
    }.map(_.toScalaVector())
  }

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = {
    /* If the underlying model has no gradient, return 0 */
    if (!predictions.head.getGradient().isDefined) {
      return None
    }
    val gradientsByPrediction: Seq[Seq[Vector[Double]]] = predictions.map(_.getGradient().get)
    val gradientsByInput: Seq[Seq[Vector[Double]]] = gradientsByPrediction.transpose
    Some(gradientsByInput.map { r =>
      r.toVector.transpose.map(_.sum / predictions.size)
    })
  }
}