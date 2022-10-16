package io.citrine.lolo.bags

import breeze.linalg.{norm, DenseMatrix, DenseVector}
import io.citrine.lolo.{PredictionResult, RegressionResult}
import org.slf4j.{Logger, LoggerFactory}

/**
  * Interface defining the return value of a [[BaggedModel]].
  *
  * This allows the implementation to depend on the number of simultaneous predictions,
  * which has performance implications.
  *
  * For background on the uncertainty calculation, see Wager, S.; Hastie, T and Efron, B. Confidence Intervals for
  * Random Forests: The Jackknife and Infinitesimal Jackknife. Journal of Machine Learning Research 15 (2014).
  */
trait BaggedPrediction[+T] extends PredictionResult[T] {

  /** The number of input rows that have been predicted on (NOT the number of bagged models). */
  def numPredictions: Int

  /** The predictions made by each of the bagged models. */
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

/** Interface for [[BaggedPrediction]]s for regression tasks with task-specific methods. */
sealed trait BaggedRegressionPrediction extends BaggedPrediction[Double] with RegressionResult

/**
  * Container with model-wise predictions at a single input point.
  *
  * Assuming a single input allows for performance optimizations and more readable code.
  * See [[MultiPointBaggedPrediction]] for a generic implementation.
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  */
case class SinglePointBaggedPrediction(
    predictions: Seq[PredictionResult[Double]],
    NibIn: Vector[Vector[Int]],
    bias: Option[Double] = None,
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false
) extends BaggedRegressionPrediction {

  override def numPredictions: Int = 1

  override def getExpected(): Seq[Double] = Seq(expected + bias.getOrElse(0.0))

  override def getStdDevMean(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      // If bootstrap is disabled, rescale is unity and treeVariance is our only option for UQ.
      // Since it's not recalibrated, it's best considered to be a confidence interval of the underlying weak learner.
      assert(rescaleRatio == 1.0)
      Some(Seq(math.sqrt(treeVariance)))
    } else {
      Some(Seq(stdDevMean))
    }
  }

  override def getStdDevObs(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      None
    } else {
      Some(Seq(stdDevObs))
    }
  }

  /**
    * For the sake of parity, we were using this method
    */
  override def getUncertainty(observational: Boolean): Option[Seq[Any]] = {
    if (observational) {
      getStdDevObs()
    } else {
      getStdDevMean()
    }
  }

  /**
    * The importances are computed as an average of bias-corrected jackknife-after-bootstrap
    * and infinitesimal jackknife methods
    *
    * @return training row scores of each prediction
    */
  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(Seq(singleScores))

  private lazy val treePredictions = predictions.map(_.getExpected().head).toArray
  private lazy val expected = treePredictions.sum / treePredictions.length
  private lazy val treeVariance = {
    assert(treePredictions.length > 1, "Bootstrap variance undefined for fewer than 2 bootstrap samples.")
    treePredictions.map(x => math.pow(x - expected, 2.0)).sum / (treePredictions.length - 1)
  }

  private lazy val stdDevMean: Double = math.sqrt(BaggedPrediction.rectifyEstimatedVariance(singleScores))

  private lazy val stdDevObs: Double = {
    rescaleRatio * math.sqrt(treeVariance)
  } ensuring (_ >= 0.0)

  private lazy val singleScores: Vector[Double] = {
    // This will be more convenient later
    val nMat = NibIn.transpose

    // Compute the Bessel-uncorrected variance of the ensemble of predicted values, then multiply by (n - 1) / (n * B)
    // We later sum over the training data, introducing a factor of n and leaving us with the expected correction term
    val correction = treeVariance * (treePredictions.length - 1.0) * (nMat.size - 1) /
      (treePredictions.length * treePredictions.length * nMat.size)

    // Loop over each of the training instances, computing its contribution to the uncertainty
    val trainingContributions = nMat.indices.toVector.map { idx =>
      // Pull the vector of the number of times this instance was used to train each tree
      val vecN = nMat(idx).toArray

      // Loop over the trees, computing the covariance for the IJ estimate and the predicted value of
      // the out-of-bag trees for the J(ackknife) estimate
      // The loops are merged for performance reasons
      var cov: Double = 0.0
      var tNot: Double = 0.0
      var tNotCount: Int = 0
      vecN.indices.foreach { jdx =>
        cov = cov + (vecN(jdx) - 1) * (treePredictions(jdx) - expected)

        if (vecN(jdx) == 0) {
          tNot = tNot + treePredictions(jdx)
          tNotCount = tNotCount + 1
        }
      }
      // Compute the infinitesimal jackknife estimate
      val varIJ = math.pow(cov / vecN.length, 2.0)

      if (tNotCount > 0) {
        // Compute the Jackknife after bootstrap estimate
        val varJ = math.pow(tNot / tNotCount - expected, 2.0) * (nMat.size - 1) / nMat.size
        // Averaged the corrected IJ and J estimates
        0.5 * (varJ + varIJ - math.E * correction)
      } else {
        // We can't compute the Jackknife after bootstrap estimate, so just correct the IJ estimate
        varIJ - correction
      }
    }
    // The uncertainty must be positive, so anything smaller than zero is noise.  Make sure that no estimated
    // uncertainty is below that noise level
    trainingContributions
  }
}

/**
  * Container with model-wise predictions and logic to compute variances and training row scores
  *
  * These calculations are implemented using matrix arithmetic to make them more performant when the number
  * of predictions is large.  This obfuscates the algorithm significantly, however.  To see what is being computed,
  * look at [[SinglePointBaggedPrediction]], which is more clear.  These two implementations are tested for consistency.
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  */
case class MultiPointBaggedPrediction(
    predictions: Seq[PredictionResult[Double]],
    NibIn: Vector[Vector[Int]],
    bias: Option[Seq[Double]] = None,
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false
) extends BaggedRegressionPrediction {

  override def getExpected(): Seq[Double] = expected.zip(biasCorrection).map(x => x._1 + x._2)

  override def getStdDevObs(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      None
    } else {
      Some(varObs.map { v => math.sqrt(v) })
    }
  }

  override def getStdDevMean(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      // If bootstrap is disabled, rescale is unity and treeVariance is our only option for UQ.
      // Since it's not recalibrated, it's best considered to be a confidence interval of the underlying weak learner.
      assert(rescaleRatio == 1.0)
      Some(varObs.map { v => math.sqrt(v) })
    } else {
      Some(stdDevMean)
    }
  }

  /**
    * For the sake of parity, we were using this method
    */
  override def getUncertainty(observational: Boolean): Option[Seq[Any]] = {
    if (observational) {
      getStdDevObs()
    } else {
      getStdDevMean()
    }
  }

  /**
    * Return IJ scores
    *
    * @return training row scores of each prediction
    */
  override def getInfluenceScores(actuals: Seq[Any]): Option[Seq[Seq[Double]]] = {
    Some(
      influences(
        expected.toVector,
        actuals.toVector.asInstanceOf[Vector[Double]],
        expectedMatrix,
        NibJMat,
        NibIJMat
      )
    )
  }

  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(scores)

  override def numPredictions: Int = expectedMatrix.length

  /* transpose to be (# training) x (# models) */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose

  /* Make a matrix of the tree-wise predictions */
  lazy val expectedMatrix: Seq[Seq[Double]] = predictions.map(p => p.getExpected()).transpose

  /* Extract the prediction by averaging over trees and adding the bias correction. */
  lazy val biasCorrection: Seq[Double] = bias.getOrElse(Seq.fill(expectedMatrix.length)(0))
  lazy val expected: Seq[Double] = expectedMatrix.map(ps => ps.sum / ps.size)

  lazy val NibJMat: DenseMatrix[Double] = BaggedPrediction.getJackknifeAfterBootstrapMatrix(Nib)

  lazy val NibIJMat: DenseMatrix[Double] = BaggedPrediction.getInfinitesimalJackknifeMatrix(Nib)

  /* This represents the variance of the estimate of the mean. */
  lazy val stdDevMean: Seq[Double] =
    variance(expected.toVector, expectedMatrix, NibJMat, NibIJMat).map { math.sqrt }

  /* This estimates the variance of predictive distribution. */
  lazy val varObs: Seq[Double] = expectedMatrix.zip(expected).map {
    case (b, y) =>
      assert(Nib.size > 1, "Bootstrap variance undefined for fewer than 2 bootstrap samples.")
      b.map { x => rescaleRatio * rescaleRatio * math.pow(x - y, 2.0) }.sum / (b.size - 1)
  }

  /* Compute the scores one prediction at a time */
  lazy val scores: Seq[Vector[Double]] = scores(expected.toVector, expectedMatrix, NibJMat, NibIJMat)
    // make sure the variance is non-negative after the stochastic correction
    .map(BaggedPrediction.rectifyImportanceScores)
    .map(_.map(math.sqrt))

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
    scores(meanPrediction, modelPredictions, NibJ, NibIJ).map { BaggedPrediction.rectifyEstimatedVariance }
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
    val predMat =
      new DenseMatrix[Double](modelPredictions.head.size, modelPredictions.size, modelPredictions.flatten.toArray)

    /* These operations are pulled out of the loop and extra-verbose for performance */
    val JMat = NibJ.t * predMat
    val JMat2 = JMat *:* JMat * ((Nib.size - 1.0) / Nib.size)
    val IJMat = NibIJ.t * predMat
    val IJMat2 = IJMat *:* IJMat
    val arg = IJMat2 + JMat2

    /* Avoid division in the loop, calculate (n - 1) / (n * B^2) */
    val prefactor = 1.0 / math.pow(modelPredictions.head.size, 2.0) * (Nib.size - 1) / Nib.size

    modelPredictions.indices
      .map { i =>
        /* Compute the first order bias correction for the variance estimators */
        val correction = prefactor * math.pow(norm(predMat(::, i) - meanPrediction(i)), 2)

        /* The correction is prediction dependent, so we need to operate on vectors */
        0.5 * (arg(::, i) - math.E * correction)
      }
      .map(_.toScalaVector)
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
    val predMat =
      new DenseMatrix[Double](modelPredictions.head.size, modelPredictions.size, modelPredictions.flatten.toArray)

    /* These operations are pulled out of the loop and extra-verbose for performance */
    val JMat = NibJ.t * predMat
    val IJMat = NibIJ.t * predMat
    val arg = IJMat + JMat

    /* Avoid division in the loop */
    val inverseSize = 1.0 / modelPredictions.head.size

    modelPredictions.indices
      .map { i =>
        /* Compute the first order bias correction for the variance estimators */
        val correction = inverseSize * norm(predMat(::, i) - meanPrediction(i))

        /* The correction is prediction dependent, so we need to operate on vectors */
        val influencePerRow: DenseVector[Double] =
          math.signum(actualPrediction(i) - meanPrediction(i)) * 0.5 * (arg(::, i) - math.E * correction)

        /* Impose a floor in case any of the variances are negative (hacked to work in breeze) */
        // val floor: Double = math.min(0, -min(variancePerRow))
        // val rezero: DenseVector[Double] = variancePerRow - floor
        // 0.5 * (rezero + abs(rezero)) + floor
        influencePerRow
      }
      .map(_.toScalaVector)
  }
}

case class BaggedClassificationPrediction[T](predictions: Seq[PredictionResult[T]]) extends BaggedPrediction[T] {
  lazy val expectedMatrix: Seq[Seq[T]] = predictions.map(p => p.getExpected()).transpose
  lazy val expected: Seq[T] = expectedMatrix.map(ps => ps.groupBy(identity).maxBy(_._2.size)._1)
  lazy val uncertainty: Seq[Map[T, Double]] =
    expectedMatrix.map(ps => ps.groupBy(identity).view.mapValues(_.size.toDouble / ps.size).toMap)

  override def numPredictions: Int = expectedMatrix.length

  override def getExpected(): Seq[T] = expected

  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Map[T, Double]]] = Some(uncertainty)
}

object BaggedPrediction {

  val logger: Logger = LoggerFactory.getLogger(getClass)

  /**
    * Generate a matrix that is useful for computing (co)variance via jackknife after bootstrap (JaB).
    * The central term of the JaB calculation (Wager et. al. 2014, equation 6) is the difference between the
    * out-of-bag prediction on a point and the mean prediction on that point. If this is written as a single sum over bags,
    * then each point has a weight -1/B when it is in-bag and weight 1/|{N_{bi}=0}| - 1/B when it is out-of-bag (B is the number of bags).
    * This matrix encodes those weights, so when it is multiplied by the (# bags) x (# predictions) prediction matrix
    * we have a matrix of the \Delta terms from equation 6.
    *
    * @param Nib The (# training) x (# bags) matrix indicating how many times each training point is used in each bag
    */
  def getJackknifeAfterBootstrapMatrix(Nib: Vector[Vector[Int]]): DenseMatrix[Double] = {
    new DenseMatrix[Double](
      Nib.head.size,
      Nib.size,
      Nib.flatMap { v =>
        val itot = 1.0 / v.size // 1/B
        val icount = 1.0 / v.count(_ == 0) // 1/|{N_{bi}=0}|
        v.map(n => if (n == 0) icount - itot else -itot)
      }.toArray
    )
  }

  /**
    * Generate a matrix that is useful for computing (co)variance via infinitesimal jackknife (IJ).
    * The central term of the IJ calculation (Wager et. al. 2014, equation 5) is the covariance between the number of
    * times a training point appears in a bag and the prediction made by that bag.
    * This matrix encodes (N - \bar{N})/B (B is the number of bags), so that when it is multiplied by the
    *  (# bags) x (# predictions) prediction matrix, we have a matrix of the covariance terms from equation 5.
    *
    * @param Nib The (# training) x (# bags) matrix indicating how many times each training point is used in each bag
    */
  def getInfinitesimalJackknifeMatrix(Nib: Vector[Vector[Int]]): DenseMatrix[Double] = {
    new DenseMatrix[Double](
      Nib.head.size,
      Nib.size,
      Nib.flatMap { v =>
        val itot = 1.0 / v.size // 1/B
        val vtot = v.sum.toDouble / (v.size * v.size) // \bar{N} / B
        v.map(n => n * itot - vtot)
      }.toArray
    )
  }

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
      logger.warn(
        s"Sum of scores was negative; using the largest score as an estimate for the variance.  Please consider increasing the ensemble size."
      )
      maxEntry
    } else {
      logger.warn(
        s"All scores were negative; using the magnitude of the smallest score as an estimate for the variance.  It is highly recommended to increase the ensemble size."
      )
      -scores.min // equivalent to math.abs(scores.min)
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
    val floor = math.abs(scores.min)
    if (floor < 0.0) {
      logger.warn(s"Some importance scores were negative; rectifying.  Please consider increasing the ensemble size.")
    }
    scores.map(math.max(floor, _))
  } ensuring (vec => vec.forall(_ >= 0.0))
}
