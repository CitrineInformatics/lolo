package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import io.citrine.lolo.{PredictionResult, RegressionResult}
import io.citrine.lolo.util.Async
import org.slf4j.{Logger, LoggerFactory}

/**
  * Interface defining the return value of a [[BaggedModel]]
  *
  * This allows the implementation to depend on the number of simultaneous predictions, which has performance
  * implications.
  */
trait BaggedResult[+T] extends PredictionResult[T] {
  def predictions: Seq[PredictionResult[T]]

  /**
    * Average the gradients from the models in the ensemble
    *
    * @return the gradient of each prediction as a vector of doubles
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = gradient

  private lazy val gradient = if (predictions.head.getGradient().isEmpty) {
    /* If the underlying model has no gradient, return None */
    None
  } else {
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
  */
case class BaggedSingleResult(
                               predictions: Seq[PredictionResult[Double]],
                               NibIn: Vector[Vector[Int]],
                               bias: Option[Double] = None,
                               rescale: Double = 1.0
                             ) extends BaggedResult[Double] with RegressionResult {
  private lazy val treePredictions: Array[Double] = predictions.map(_.getExpected().head).toArray

  /**
    * Return the ensemble average or maximum vote
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = Seq(expected)

  private lazy val expected = treePredictions.sum / treePredictions.length
  private lazy val treeVariance: Double = treePredictions.map(x => Math.pow(x - expected, 2.0)).sum / treePredictions.length

  override def getStdDevObs(): Option[Seq[Double]] = Some(Seq(treeVariance))

  override def getStdDev(): Option[Seq[Double]] = Some(Seq(scalarUncertainty))

  private lazy val scalarUncertainty: Double = {
    // make sure the variance is non-negative after the stochastic correction
    val rectified = BaggedResult.rectifyEstimatedVariance(singleScores)
    Math.sqrt(rectified * Math.pow(rescale, 2.0) + Math.pow(bias.getOrElse(0.0), 2.0))
  } ensuring(_ >= 0.0)


  /**
    * The importances are computed as an average of bias-corrected jackknife-after-bootstrap
    * and infinitesimal jackknife methods
    *
    * @return training row scores of each prediction
    */
  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(Seq(singleScores))

  private lazy val singleScores: Vector[Double] = {
    // Compute the variance of the ensemble of predicted values and divide by the size of the ensemble an extra time
    val varT = treeVariance / treePredictions.length

    // This will be more convenient later
    val nMat = NibIn.transpose

    // Loop over each of the training instances, computing its contribution to the uncertainty
    val trainingContributions = nMat.indices.toVector.map { idx =>
      // Pull the vector of the number of times this instance was used to train each tree
      val vecN = nMat(idx).toArray
      val nTot = vecN.sum

      // Loop over the trees, computing the covariance for the IJ estimate and the predicted value of
      // the out-of-bag trees for the J(ackknife) estimate
      // The loops are merged for performance reasons
      var cov: Double = 0.0
      var tNot: Double = 0.0
      var tNotCount: Int = 0
      vecN.indices.foreach { jdx =>
        cov = cov + (vecN(jdx) - nTot) * (treePredictions(jdx) - expected)

        if (vecN(jdx) == 0) {
          tNot = tNot + treePredictions(jdx)
          tNotCount = tNotCount + 1
        }
      }
      // Compute the infinitessimal jackknife estimate
      val varIJ = Math.pow(cov / vecN.size, 2.0)

      if (tNotCount > 0) {
        // Compute the Jackknife after bootstrap estimate
        val varJ = Math.pow(tNot / tNotCount - expected, 2.0) * (nMat.size - 1) / nMat.size
        // Compute the sum of the corrections to the IJ and J estimates
        val correction = Math.E * varT
        // Averaged the correct IJ and J estimates
        0.5 * (varJ + varIJ - correction)
      } else {
        // We can't compute the Jackknife after bootstrap estimate, so just correct the IJ estimate
        val correction = varT
        varIJ - correction
      }
    }
    // The uncertainty must be positive, so anything smaller than zero is noise.  Make sure that no estimated
    // uncertainty is below that noise level
    trainingContributions
  }
}

case class BaggedClassificationResult(
                                  predictions: Seq[PredictionResult[Any]]
                                ) extends BaggedResult[Any] {
  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose

  lazy val expected: Seq[Any] = expectedMatrix.map(ps => ps.groupBy(identity).maxBy(_._2.size)._1).seq
  lazy val uncertainty: Seq[Map[Any, Double]] = expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size.toDouble / ps.size))

  /**
   * Return the majority vote vote
   *
   * @return expected value of each prediction
   */
  override def getExpected(): Seq[Any] = expected

  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = Some(uncertainty)
}

/**
  * Container with model-wise predictions and logic to compute variances and training row scores
  *
  * These calculations are implemented using matrix arithmetic to make them more performant when the number
  * of predictions is large.  This obfuscates the algorithm significantly, however.  To see what is being computed,
  * look at [[BaggedSingleResult]], which is more clear.  These two implementations are tested for consistency.
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  */
case class BaggedMultiResult(
                         predictions: Seq[PredictionResult[Double]],
                         NibIn: Vector[Vector[Int]],
                         bias: Option[Seq[Double]] = None,
                         rescale: Double = 1.0
                       ) extends BaggedResult[Double] with RegressionResult {

  /**
    * Return the ensemble average
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = expected

  override def getStdDevObs(): Option[Seq[Double]] = Some{
    expectedMatrix.asInstanceOf[Seq[Seq[Double]]].zip(expected.asInstanceOf[Seq[Double]]).map{case (b, y) =>
      b.map{x => Math.pow(x - y, 2.0)}.sum / b.size
    }
  }

  override def getStdDev(): Option[Seq[Double]] = Some(uncertainty)

  /**
    * Return IJ scores
    *
    * @return training row scores of each prediction
    */
  override def getInfluenceScores(actuals: Seq[Any]): Option[Seq[Seq[Double]]] = {
    Some(influences(
      expected.asInstanceOf[Seq[Double]].toVector,
      actuals.toVector.asInstanceOf[Vector[Double]],
      expectedMatrix.asInstanceOf[Seq[Seq[Double]]],
      NibJMat,
      NibIJMat
    ))
  }

  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(scores)

  /* Subtract off 1 to make correlations easier; transpose to be prediction-wise */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  /* Make a matrix of the tree-wise predictions */
  lazy val expectedMatrix: Seq[Seq[Double]] = predictions.map(p => p.getExpected()).transpose

  /* Extract the prediction by averaging for regression, taking the most popular response for classification */
  lazy val expected: Seq[Double] = expectedMatrix.map(ps => ps.sum / ps.size)

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
      v.map(n => n * itot - vtot)
    }.toArray
  )

  /* Compute the uncertainties one prediction at a time */
  lazy val uncertainty: Seq[Double] = {
    val sigma2: Seq[Double] = variance(expected.asInstanceOf[Seq[Double]].toVector, expectedMatrix, NibJMat, NibIJMat)
    val rescale2 = rescale * rescale
    sigma2.zip(bias.getOrElse(Seq.fill(expected.size)(0.0))).map { case (variance, b) => Math.sqrt(b * b + variance * rescale2) }
  }

  /* Compute the scores one prediction at a time */
  lazy val scores: Seq[Vector[Double]] = scores(expected.toVector, expectedMatrix, NibJMat, NibIJMat)
        // make sure the variance is non-negative after the stochastic correction
        .map(BaggedResult.rectifyImportanceScores)
        .map(_.map(Math.sqrt))

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
    scores(meanPrediction, modelPredictions, NibJ, NibIJ).map{BaggedResult.rectifyEstimatedVariance}
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

    modelPredictions.indices.map { i =>
      Async.canStop()
      /* Compute the first order bias correction for the variance estimators */
      val correction = Math.pow(inverseSize * norm(predMat(::, i) - meanPrediction(i)), 2)

      /* The correction is prediction dependent, so we need to operate on vectors */
      0.5 * (arg(::, i) - Math.E * correction)
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

    modelPredictions.indices.map { i =>
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
}

object BaggedResult {

  val logger: Logger = LoggerFactory.getLogger(getClass)

  /**
   * Make sure the variance is non-negative
   *
   * The monte carlo bias correction is itself stochastic, so let's make sure the result is positive
   *
   * If the sum is positive, then great!  We're done.
   *
   * If the sum is <= 0.0, then the actual variance is likely quite small.  We know the variance should be at
   * least as large as the largest importance, since at least one training point will be important.
   * Therefore, let's just take the maximum importance, which should be a reasonable lower-bound of the variance.
   * Note that we could also sum the non-negative scores, but that could be biased upwards.
   *
   * If all of the scores are negative (which happens infrequently for very small ensembles), then we just need a scale.
   * The largest scale is the largest magnitude score, which is the absolute value of the minimum score.  When this
   * happens, then a larger ensemble should really be used!
   *
   * If all of the treePredictions are zero, then this will return zero.
   *
   * @param scores the monte-carlo corrected importance scores
   * @return A non-negative estimate of the variance
   */
  def rectifyEstimatedVariance(scores: Seq[Double]): Double = {
    val rawSum = scores.sum
    lazy val maxEntry = scores.max

    if (rawSum > 0) {
      rawSum
    } else if (maxEntry > 0) {
      // If the sum is negative,
      logger.warn(s"Sum of scores was negative; using the largest score as an estimate for the variance.  Please consider increasing the ensemble size.")
      maxEntry
    } else {
      logger.warn(s"All scores were negative; using the magnitude of the smallest score as an estimate for the variance.  It is highly recommended to increase the ensemble size.")
      - scores.min // equivalent to Math.abs(scores.min)
    }
  } ensuring (_ >= 0.0)

  /**
   * Make sure the scores are each non-negative
   *
   * The monte carlo bias correction is itself stochastic, so let's make sure the result is positive.
   * If the score was statistically consistent with zero, then we might subtract off the entire bias correction,
   * which results in the negative value.  Therefore, we can use the magnitude of the minimum as an estimate of the noise
   * level, and can simply set that as a floor.
   *
   * If all of the treePredictions are zero, then this will return a vector of zero
   *
   * @param scores the monte-carlo corrected importance scores
   * @return a vector of non-negative bias corrected scores
   */
  def rectifyImportanceScores(scores: Vector[Double]): Vector[Double] = {
    // this is a lower-bound on the noise level; note that it is strictly smaller than the correction
    val floor = Math.abs(scores.min)

    if (floor < 0.0) {
      logger.warn(s"Some importance scores were negative; rectifying.  Please consider increasing the ensemble size.")
    }
    scores.map(Math.max(floor, _))
  } ensuring (vec => vec.forall(_ >= 0.0))
}
