package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix, DenseVector, min, norm, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Poisson
import io.citrine.lolo.results.{PredictionResult, TrainingResult, hasFeatureImportance, hasLoss, hasPredictedVsActual}
import io.citrine.lolo.{Learner, Model}

import scala.collection.parallel.immutable.ParSeq

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data
  *
  * Created by maxhutch on 11/14/16.
  *
  * @param method  learner to train each model in the ensemble
  * @param numBags number of models in the ensemble
  */
class Bagger(
              method: Learner,
              var numBags: Int = -1,
              val useJackknife: Boolean = true,
              biasLearner: Option[Learner] = None
            ) extends Learner {

  override def setHypers(moreHypers: Map[String, Any]): Bagger.this.type = {
    method.setHypers(moreHypers)
    super.setHypers(moreHypers)
  }

  override var hypers: Map[String, Any] = Map()

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

    /* Learn the actual models */
    val models = (0 until actualBags).par.map { i =>
      method.train(trainingData.toVector, Some(Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)))
    }

    val importances: Array[Double] = models.map(base => base.asInstanceOf[hasFeatureImportance].getFeatureImportance()).reduce { (v1, v2) =>
      v1.zip(v2).map(p => p._1 + p._2)
    }.map(_ / models.size)

    /* Wrap the models in a BaggedModel object */
    if (biasLearner.isEmpty) {
      new BaggedTrainingResult(models.map(_.getModel()), hypers, importances, Nib, trainingData, useJackknife)
    } else {
      val baggedModel = new BaggedModel(models.map(_.getModel()), Nib, useJackknife)
      val baggedRes = baggedModel.transform(trainingData.map(_._1))
      val biasTraining = trainingData.zip(
        baggedRes.getExpected().zip(baggedRes.getUncertainty().get)
      ).map { case ((f, a), (p, u)) =>
        // Math.E is only statistically correct.  It should be actualBags / Nib.transpose(i).count(_ == 0)
        // Or, better yet, filter the bags that don't include the training example
        val bias = Math.max(Math.E * Math.abs(p.asInstanceOf[Double] - a.asInstanceOf[Double]) - u.asInstanceOf[Double], 0.0)
        (f, bias)
      }
      val biasModel = biasLearner.get.train(biasTraining).getModel()
      new BaggedTrainingResult(models.map(_.getModel()), hypers, importances, Nib, trainingData, useJackknife, Some(biasModel))
    }
  }
}

@SerialVersionUID(999L)
class BaggedTrainingResult(
                            models: ParSeq[Model[PredictionResult[Any]]],
                            hypers: Map[String, Any],
                            featureImportance: Array[Double],
                            Nib: Vector[Vector[Int]],
                            trainingData: Seq[(Vector[Any], Any)],
                            useJackknife: Boolean,
                            biasModel: Option[Model[PredictionResult[Any]]] = None
                          )
  extends TrainingResult with hasPredictedVsActual with hasLoss with hasFeatureImportance {
  lazy val NibT = Nib.transpose
  lazy val model = new BaggedModel(models, Nib, useJackknife, biasModel)
  lazy val rep = trainingData.head._2
  lazy val predictedVsActual = trainingData.zip(NibT).map { case ((f, l), nb) =>
    val predicted = rep match {
      case x: Double => models.zip(nb).filter(_._2 == 0).map(_._1.transform(Seq(f)).getExpected().head.asInstanceOf[Double]).sum / nb.count(_ == 0)
      case x: Any => models.zip(nb).filter(_._2 == 0).map(_._1.transform(Seq(f)).getExpected().head).groupBy(identity).maxBy(_._2.size)._1
    }
    (f, predicted, l)
  }

  lazy val loss = rep match {
    case x: Double => Math.sqrt(predictedVsActual.map(d => Math.pow(d._2.asInstanceOf[Double] - d._3.asInstanceOf[Double], 2)).sum / predictedVsActual.size)
    case x: Any =>
      val labels = predictedVsActual.map(_._3).distinct
      val index = labels.zipWithIndex.toMap
      val numLabels = labels.size
      val confusionMatrix = DenseMatrix.zeros[Int](numLabels, numLabels)
      predictedVsActual.foreach(p => confusionMatrix(index(p._2), index(p._3)) += 1)
      val f1scores = labels.indices.map { i =>
        val actualPositive: Double = sum(confusionMatrix(::, i))
        val predictedPositive: Double = sum(confusionMatrix(i, ::))
        val precision = if (predictedPositive > 0) confusionMatrix(i, i) / predictedPositive else 1.0
        val recall = if (actualPositive > 0) confusionMatrix(i, i) / actualPositive else 1.0
        2.0 * precision * recall / (precision + recall) * actualPositive
      }
      f1scores.sum / trainingData.size
  }

  /**
    * Average the importances across the ensemble of models
    *
    * @return feature importances as an array of doubles
    */
  override def getFeatureImportance(): Array[Double] = featureImportance

  override def getModel(): BaggedModel = model

  override def getPredictedVsActual(): Seq[(Vector[Any], Any, Any)] = predictedVsActual

  override def getLoss(): Double = loss

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
  override def getScores(): Option[Seq[Seq[Double]]] = Some(scores)

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
    /* Stick the predictions in a breeze matrix */
    val predMat = new DenseMatrix[Double](modelPredictions.head.size, modelPredictions.size, modelPredictions.flatten.toArray)

    /* These operations are pulled out of the loop and extra-verbose for performance */
    val JMat = NibJ.t * predMat
    val JMat2 = JMat :* JMat * ((Nib.size - 1.0) / Nib.size)
    val IJMat = NibIJ.t * predMat
    val IJMat2 = IJMat :* IJMat
    val arg = IJMat2 + JMat2

    /* Avoid division in the loop */
    val inverseSize = 1.0 / modelPredictions.head.size

    (0 until modelPredictions.size).map{ i =>
      /* Compute the first order bias correction for the variance estimators */
      val correction = Math.pow(inverseSize * norm(predMat(::, i) - meanPrediction(i)), 2)

      /* The correction is prediction dependent, so we need to operate on vectors */
      val variancePerRow: DenseVector[Double] = 0.5 * ( arg(::, i) - Math.E * correction)

      /* Impose a floor in case any of the variances are negative (hacked to work in breeze) */
      val floor: Double = Math.min(0, -min(variancePerRow))
      val rezero: DenseVector[Double] = variancePerRow - floor
      0.5 * sum(rezero + abs(rezero)) + floor
    }
  }

  /**
    * Compute the IJ training row scores for a prediction
    *
    * @param meanPrediction   across the models
    * @param modelPredictions predictions of each model
    * @param Nib              sample matrix
    * @return the score of each training row as a vector of doubles
    */
  def scores(meanPrediction: Double, modelPredictions: Seq[Double], Nib: Seq[Vector[Int]]): Vector[Double] = {
    val diff = modelPredictions.map(_ - meanPrediction)

    /* Compute the IJ score for each row */
    Nib.map { v =>
      Math.abs(v.zip(diff).map(p2 => p2._1 * p2._2).sum / modelPredictions.size)
    }.toVector
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