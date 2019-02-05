package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix, DenseVector, min, norm}
import breeze.numerics.abs
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.util.Async

trait BaggedResult extends  PredictionResult[Any]

/**
  * Container with model-wise predictions and logic to compute variances and training row scores
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  * @param repInput    representative input
  */
class BaggedMultiResult(
                    predictions: Seq[PredictionResult[Any]],
                    NibIn: Vector[Vector[Int]],
                    useJackknife: Boolean,
                    bias: Option[Seq[Double]] = None,
                    repInput: Vector[Any],
                    rescale: Double = 1.0
                  ) extends BaggedResult {

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
      sigma2.zip(bias.getOrElse(Seq.fill(expected.size)(0.0))).map(p => Math.sqrt(p._2 * p._2 + p._1)).map(_ * rescale)
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
      val correction = inverseSize * norm(predMat(::, i) - meanPrediction(i))

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

/**
  * Container with model-wise predictions and logic to compute variances and training row scores
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  * @param repInput    representative input
  */
class BaggedSingleResult(
                    predictions: Seq[PredictionResult[Any]],
                    NibIn: Vector[Vector[Int]],
                    useJackknife: Boolean,
                    bias: Option[Double] = None,
                    repInput: Vector[Any],
                    rescale: Double = 1.0
                  ) extends BaggedResult {

  /**
    * Return the ensemble average or maximum vote
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = Seq(expected)

  /**
    * Return jackknife-based variance estimates
    *
    * @return uncertainty of each prediction
    */
  override def getUncertainty(): Option[Seq[Any]] = Some(Seq(uncertainty))

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

  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(Seq(scores))

  /* Subtract off 1 to make correlations easier; transpose to be prediction-wise */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  /* Make a matrix of the tree-wise predictions */
  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose

  /* For checking the type of the prediction */
  lazy val rep: Any = expectedMatrix.head.head

  /* Extract the prediction by averaging for regression, taking the most popular response for classification */
  lazy val expected = rep match {
    case x: Double => expectedMatrix.map(ps => ps.asInstanceOf[Seq[Double]].sum / ps.size).head
    case x: Any => expectedMatrix.map(ps => ps.groupBy(identity).maxBy(_._2.size)._1).seq.head
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
      val sigma2: Double = if (useJackknife) {
        variance(expected.asInstanceOf[Seq[Double]].toVector, expectedMatrix.asInstanceOf[Seq[Seq[Double]]], NibJMat, NibIJMat).head
      } else {
        0.0
      }
      sigma2 + Math.pow(bias.getOrElse(0.0), 2.0) * rescale
    case x: Any =>
      expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size.toDouble / ps.size)).head
  }

  /* Compute the scores one prediction at a time */
  lazy val scores: Vector[Double] = rep match {
    case x: Double =>
      scores(expected.asInstanceOf[Seq[Double]].toVector, expectedMatrix.asInstanceOf[Seq[Seq[Double]]], NibJMat, NibIJMat).map(_.map(Math.sqrt)).head
    case x: Any => Vector.fill(Nib.size)(0.0)
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
      val correction = inverseSize * norm(predMat(::, i) - meanPrediction(i))

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
