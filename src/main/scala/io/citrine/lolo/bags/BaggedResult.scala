package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import io.citrine.lolo.bags.CorrelationMethods.{Bootstrap, CorrelationMethod, FromTraining, Jackknife, Trivial, JackknifeExplicit}
import io.citrine.lolo.stats.StatsUtils
import io.citrine.lolo.{MultiTaskModelPredictionResult, ParallelModelsPredictionResult, PredictionResult, RegressionResult}
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

  /** The number of inputs that have been predicted on (NOT the number of bagged models). */
  def numPredictions: Int

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
  * Container with model-wise predictions at a single input point.
  * Assuming a single input allows for performance optimizations and more readable code.
  * See [[MultiPredictionBaggedResult]] for a generic implementation.
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  */
case class SinglePredictionBaggedResult(
                                         predictions: Seq[PredictionResult[Double]],
                                         NibIn: Vector[Vector[Int]],
                                         bias: Option[Double] = None,
                                         rescale: Double = 1.0,
                                         disableBootstrap: Boolean = false
                                       ) extends BaggedResult[Double] with RegressionResult {
  private lazy val treePredictions: Array[Double] = predictions.map(_.getExpected().head).toArray

  override def numPredictions: Int = 1

  /**
    * Return the ensemble average or maximum vote
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = Seq(expected + bias.getOrElse(0.0))

  private lazy val expected = treePredictions.sum / treePredictions.length
  private lazy val treeVariance: Double = {
    assert(treePredictions.length > 1, "Bootstrap variance undefined for fewer than 2 bootstrap samples.")
    treePredictions.map(x => Math.pow(x - expected, 2.0)).sum / (treePredictions.length - 1)
  }

  override def getStdDevMean(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      // If bootstrap is disabled, rescale is unity and treeVariance is our only option for UQ.
      // Since it's not recalibrated, it's best considered to be a confidence interval of the underlying weak learner.
      assert(rescale == 1.0)
      Some(Seq(Math.sqrt(treeVariance)))
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

  private lazy val stdDevMean: Double = Math.sqrt(BaggedResult.rectifyEstimatedVariance(singleScores))

  private lazy val stdDevObs: Double = {
    rescale * Math.sqrt(treeVariance)
  } ensuring(_ >= 0.0)

  /**
    * The importances are computed as an average of bias-corrected jackknife-after-bootstrap
    * and infinitesimal jackknife methods
    *
    * @return training row scores of each prediction
    */
  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(Seq(singleScores))

  private lazy val singleScores: Vector[Double] = {
    // Compute the Bessel-uncorrected variance of the ensemble of predicted values,
    // and then divide by the size of the ensemble an extra time
    val varT = treeVariance * (treePredictions.length - 1.0) / (treePredictions.length * treePredictions.length)

    // This will be more convenient later
    val nMat = NibIn.transpose

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
  val predictionEnsemble = predictions.map{ p => p.getExpected() }
  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose

  lazy val expected: Seq[Any] = expectedMatrix.map(ps => ps.groupBy(identity).maxBy(_._2.size)._1).seq
  lazy val uncertainty: Seq[Map[Any, Double]] = expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size.toDouble / ps.size).toMap)

  override def numPredictions: Int = expectedMatrix.length

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
  * look at [[SinglePredictionBaggedResult]], which is more clear.  These two implementations are tested for consistency.
  *
  * @param predictions for each constituent model
  * @param NibIn       the sample matrix as (N_models x N_training)
  * @param bias        model to use for estimating bias
  */
case class MultiPredictionBaggedResult(
                                        predictions: Seq[PredictionResult[Double]],
                                        NibIn: Vector[Vector[Int]],
                                        bias: Option[Seq[Double]] = None,
                                        rescale: Double = 1.0,
                                        disableBootstrap: Boolean = false
                                      ) extends BaggedResult[Double] with RegressionResult {

  /**
    * Return the ensemble average
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = expected.zip(biasCorrection).map(x => x._1 + x._2)

  override def getStdDevObs(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      None
    } else {
      Some(varObs.map { v => Math.sqrt(v) })
    }
  }

  override def getStdDevMean(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      // If bootstrap is disabled, rescale is unity and treeVariance is our only option for UQ.
      // Since it's not recalibrated, it's best considered to be a confidence interval of the underlying weak learner.
      assert(rescale == 1.0)
      Some(varObs.map{v => Math.sqrt(v)})
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
    Some(influences(
      expected.asInstanceOf[Seq[Double]].toVector,
      actuals.toVector.asInstanceOf[Vector[Double]],
      expectedMatrix.asInstanceOf[Seq[Seq[Double]]],
      NibJMat,
      NibIJMat
    ))
  }

  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(scores)

  override def numPredictions: Int = expectedMatrix.length

  /* Subtract off 1 to make correlations easier; transpose to be prediction-wise */
  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  /* Make a matrix of the tree-wise predictions */
  lazy val expectedMatrix: Seq[Seq[Double]] = predictions.map(p => p.getExpected()).transpose

  /* Extract the prediction by averaging over trees and adding the bias correction. */
  lazy val biasCorrection: Seq[Double] = bias.getOrElse(Seq.fill(expectedMatrix.length)(0))
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

  /* This represents the variance of the estimate of the mean. */
  lazy val stdDevMean: Seq[Double] = variance(expected.asInstanceOf[Seq[Double]].toVector, expectedMatrix, NibJMat, NibIJMat).map{Math.sqrt}

  /* This estimates the variance of predictive distribution. */
  lazy val varObs: Seq[Double] = expectedMatrix.asInstanceOf[Seq[Seq[Double]]].zip(expected.asInstanceOf[Seq[Double]]).map { case (b, y) =>
    assert(Nib.size > 1, "Bootstrap variance undefined for fewer than 2 bootstrap samples.")
    b.map { x => rescale * rescale * Math.pow(x - y, 2.0) }.sum / (b.size - 1)
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
    val JMat2 = JMat *:* JMat * ((Nib.size - 1.0) / Nib.size)
    Async.canStop()
    val IJMat = NibIJ.t * predMat
    Async.canStop()
    val IJMat2 = IJMat *:* IJMat
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

/**
  * Container with model-wise predictions for each label and the machinery to compute (co)variance.
  *
  * @param baggedPredictions  bagged prediction results for each label
  * @param realLabels         a boolean sequence indicating which labels are real-valued
  * @param NibIn              the sampling matrix as (# bags) x (# training)
  */
case class MultiTaskBaggedResult(
                                  baggedPredictions: Seq[BaggedResult[Any]],
                                  realLabels: Seq[Boolean],
                                  NibIn: Vector[Vector[Int]],
                                  trainingLabels: Seq[Seq[Any]],
                                  trainingWeights: Seq[Double]
                                ) extends BaggedResult[Seq[Any]] with MultiTaskModelPredictionResult {

  lazy val Nib: Vector[Vector[Int]] = NibIn.transpose.map(_.map(_ - 1))

  /* This matrix is used to compute the jackknife covariance */
  lazy val NibJMat = new DenseMatrix[Double](Nib.head.size, Nib.size,
    Nib.flatMap { v =>
      val itot = 1.0 / v.size
      val icount = 1.0 / v.count(_ == -1.0)
      v.map(n => if (n == -1) icount - itot else -itot)
    }.toArray
  )

  /* This matrix is used to compute the IJ covariance */
  lazy val NibIJMat = new DenseMatrix[Double](Nib.head.size, Nib.size,
    Nib.flatMap { v =>
      val itot = 1.0 / v.size
      val vtot = v.sum.toDouble / (v.size * v.size)
      v.map(n => n * itot - vtot)
    }.toArray
  )

  override def numPredictions: Int = baggedPredictions.head.numPredictions

  override def getExpected(): Seq[Seq[Any]] = baggedPredictions.map(_.getExpected()).transpose

  override def predictions: Seq[PredictionResult[Seq[Any]]] = baggedPredictions
    .map(_.predictions.map(_.getExpected()))
    .transpose
    .map(x => new ParallelModelsPredictionResult(x.transpose))

  // For each prediction, the uncertainty is a sequence of optional entries, one for each label.
  override def getUncertainty(observational: Boolean = true): Option[Seq[Seq[Option[Any]]]] = {
    Some(baggedPredictions.map { predictionResult =>
      predictionResult.getUncertainty(observational) match {
        case Some(value) => value.map(Some(_))
        case None => Seq.fill(numPredictions)(None)
      }
    }.transpose)
  }

  /** For now, call the trivial implementation. */
  override def getUncertaintyCorrelation(i: Int, j: Int): Option[Seq[Double]] = getUncertaintyCorrelationTrivial(i, j)

  /**
    * Calculate the uncertainty correlation using one of a variety of methods.
    * This method ensures that the two labels are distinct and both real-valued, and then dispatches the calculation
    * to the specified method.
    *
    * @param i      index of the first label
    * @param j      index of the second label
    * @param method method by which to calculate correlation
    * @return       optional sequence of correlation coefficients between specified labels for each prediction
    */
  def getUncertaintyCorrelationBuffet(i: Int, j: Int, method: CorrelationMethod): Option[Seq[Double]] = {
    (realLabels(i), realLabels(j)) match {
      case (true, true) if i == j => Some(Seq.fill(numPredictions)(1.0))
      case (true, true) => method match {
        case Trivial => getUncertaintyCorrelationTrivial(i, j)
        case FromTraining => getUncertaintyCorrelationTraining(i, j)
        case Bootstrap => getUncertaintyCorrelationBootstrap(i, j)
        case Jackknife => getUncertaintyCorrelationJackknife(i, j)
        case JackknifeExplicit => getUncertaintyCorrelationJackknifeExplicit(i, j)
      }
      case _: Any => None
    }
  }

  /** Uncertainty correlation is always 0.0 */
  private def getUncertaintyCorrelationTrivial(i: Int, j: Int): Option[Seq[Double]] = {
    Some(Seq.fill(numPredictions)(0.0))
  }

  /** Uncertainty correlation is always equal to the correlation coefficient computed over the training data. */
  private def getUncertaintyCorrelationTraining(i: Int, j: Int): Option[Seq[Double]] = {
      val yI = trainingLabels(i).asInstanceOf[Seq[Double]]
      val yJ = trainingLabels(j).asInstanceOf[Seq[Double]]
      val rho = StatsUtils.correlation(yI, yJ, Some(trainingWeights))
      Some(Seq.fill(numPredictions)(rho))
  }

  /** Uncertainty correlation is equal to the correlation coefficient of the bootstrap predictions. */
  private def getUncertaintyCorrelationBootstrap(i: Int, j: Int): Option[Seq[Double]] = {
      // make (# predictions) x (# bags) prediction matrices for each label
      val baggedPredictionsI = baggedPredictions(i).predictions.map(_.getExpected()).transpose.asInstanceOf[Seq[Seq[Double]]]
      val baggedPredictionsJ = baggedPredictions(j).predictions.map(_.getExpected()).transpose.asInstanceOf[Seq[Seq[Double]]]
      // Note that this does not take bias model into account
      Some(baggedPredictionsI.zip(baggedPredictionsJ).map { case (bagsI, bagsJ) =>
        StatsUtils.correlation(bagsI, bagsJ)
      })
  }

  /** The covariance is estimated using a combination of the infinitesimal jackknife and jackknife-after-bootstrap
    * techniques from Wager et. al. (2014), adapted from variance to covariance. The uncertainty correlation is the
    * covariance divided by the geometric mean of the variances.
    * This implementation uses matrix arithmetic for performance reasons, but it is difficult to decipher.
    *
    */
  private def getUncertaintyCorrelationJackknife(i: Int, j: Int): Option[Seq[Double]] = {
    val sigmaIOption = baggedPredictions(i).getUncertainty(observational = false).map(_.asInstanceOf[Seq[Double]])
    val sigmaJOption = baggedPredictions(j).getUncertainty(observational = false).map(_.asInstanceOf[Seq[Double]])
    (sigmaIOption, sigmaJOption) match {
      case (Some(sigmaISeq), Some(sigmaJSeq)) =>
        // make (# predictions) x (# bags) prediction matrices for each label
        val baggedPredictionsI = baggedPredictions(i).predictions.map(_.getExpected()).transpose.asInstanceOf[Seq[Seq[Double]]]
        val baggedPredictionsJ = baggedPredictions(j).predictions.map(_.getExpected()).transpose.asInstanceOf[Seq[Seq[Double]]]
        val numBags = baggedPredictionsI.head.size
        val numPredictions = baggedPredictionsI.size
        // mean value for each prediction
        val expectedI = baggedPredictionsI.map(ps => StatsUtils.mean(ps))
        val expectedJ = baggedPredictionsJ.map(ps => StatsUtils.mean(ps))
        // Stick the individual predictions into Breeze matrices
        val predMatI = new DenseMatrix[Double](numBags, numPredictions, baggedPredictionsI.flatten.toArray)
        val predMatJ = new DenseMatrix[Double](numBags, numPredictions, baggedPredictionsJ.flatten.toArray)
        // Perform the jackknife calculation on the predictions for labels i and j, then multiply them together
        // with factor of (n - 1) / n, where n is the number of training data
        val JMatI = NibJMat.t * predMatI
        val JMatJ = NibJMat.t * predMatJ
        val JMat2: DenseMatrix[Double] = JMatI *:* JMatJ * ((Nib.size - 1.0) / Nib.size)
        // Perform the infinitesimal jackknife calculation on the predictions for labels i and j,then multiply them together
        val IJMatI = NibIJMat.t * predMatI
        val IJMatJ = NibIJMat.t * predMatJ
        val IJMat2: DenseMatrix[Double] = IJMatI *:* IJMatJ
        // Add J(ackknife) and IJ covariance terms together
        val totalCovarianceTerm = JMat2 + IJMat2
        // Calculate 1/B^2 once, to avoid doing it in the loop
        val inverseSize2 = math.pow(1.0 / numBags, 2.0)
        // Loop over predictions, and for each one calculate the bias correction term and subtract it from each covariance term
        // The resulting structure has size (# predictions) x (# bags)
        val scores = baggedPredictionsI.indices.map { k =>
          val correctionI = predMatI(::, k) - expectedI(k)
          val correctionJ = predMatJ(::, k) - expectedJ(k)
          val correction = correctionI.dot(correctionJ) * inverseSize2
          0.5 * (totalCovarianceTerm(::, k) - math.E * correction)
        }.map(_.toScalaVector())
        // For each prediction, rectify the covariance scores to compute correlation
        Some(
          scores.lazyZip(sigmaISeq).lazyZip(sigmaJSeq).map { (trainingContributions, sigmaI, sigmaJ) =>
            BaggedResult.rectifyCorrelationScores(trainingContributions, sigmaI, sigmaJ)
          }
        )
      case _ => None
    }
  }

  private def getUncertaintyCorrelationJackknifeExplicit(i: Int, j: Int): Option[Seq[Double]] = {
    val sigmaIOption = baggedPredictions(i).getUncertainty(observational = false).map(_.asInstanceOf[Seq[Double]])
    val sigmaJOption = baggedPredictions(j).getUncertainty(observational = false).map(_.asInstanceOf[Seq[Double]])
    (sigmaIOption, sigmaJOption) match {
      case (Some(sigmaISeq), Some(sigmaJSeq)) =>
        Some(Seq.tabulate(numPredictions) { predIndex => // loop over each prediction input
          // Get the individual tree predictions for this input
          val treePredictionsI = baggedPredictions(i).predictions.map(_.getExpected()(predIndex)).asInstanceOf[Seq[Double]]
          val treePredictionsJ = baggedPredictions(j).predictions.map(_.getExpected()(predIndex)).asInstanceOf[Seq[Double]]
          val expectedI = StatsUtils.mean(treePredictionsI)
          val expectedJ = StatsUtils.mean(treePredictionsJ)
          val covarTrees = StatsUtils.covariance(treePredictionsI, treePredictionsJ)
          val nMat = NibIn.transpose  // transpose to (# training) x (# bags), for convenience

          // Loop over the training instances, computing each one's contribution to covariance.
          val trainingContributions = nMat.indices.toVector.map { trainIndex =>
            val vecN = nMat(trainIndex).toArray  // number of times this instance was used to train each tree
            var covI: Double = 0.0  // IJ covariance between training count and prediction on label i
            var covJ: Double = 0.0  // IJ covariance between training count and prediction on label j
            var tNotI: Double = 0.0  // total out-of-bag predictions on label i
            var tNotJ: Double = 0.0  // total out-of-bag predictions on label j
            var tNotCount: Int = 0  // total number of trees that do not include this training instance

            vecN.indices.foreach { bagIndex =>
              covI = covI + (vecN(bagIndex) - 1) * (treePredictionsI(bagIndex) - expectedI)
              covJ = covJ + (vecN(bagIndex) - 1) * (treePredictionsJ(bagIndex) - expectedJ)

              if (vecN(bagIndex) == 0) {
                tNotI = tNotI + treePredictionsI(bagIndex)
                tNotJ = tNotJ + treePredictionsJ(bagIndex)
                tNotCount = tNotCount + 1
              }
            }

            val covarIJ = covI * covJ / math.pow(vecN.size, 2.0)  // Infiniestimal Jackknife (IJ) term

            // Average IJ term with J term (if applicable) and subtract bias correction term
            if (tNotCount > 0) {
              // Jackknife (J) term
              val covarJ = (tNotI / tNotCount - expectedI) * (tNotJ / tNotCount - expectedJ) * (nMat.size - 1) / nMat.size
              0.5 * (covarJ + covarIJ - Math.E * covarTrees / vecN.size)
            } else {
              covarIJ - covarTrees / vecN.size
            }
          }
          BaggedResult.rectifyCorrelationScores(trainingContributions, sigmaISeq(predIndex), sigmaJSeq(predIndex))
        })
    }
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

  /**
    * Calculate the correlation coefficient given a sequence of covariance scores for each training point.
    * In theory, the covariance is the sum of the individual covariance scores. But because each covariance score
    * is noisy, the resulting sum can be larger in absolute value than `sigmaX * sigmaY`, which would translate into
    * a correlation coefficient that is outside [-1.0, 1.0].
    *
    * @param covarianceScores sequence of Monte Carlo corrected covariance contributions for each training point
    * @param sigmaX           uncertainty in first label
    * @param sigmaY           uncertainty in second label
    * @return                 correlation coefficient, guaranteed to be between -1.0 and 1.0
    */
  def rectifyCorrelationScores(covarianceScores: Vector[Double], sigmaX: Double, sigmaY: Double): Double = {
    require(sigmaX >= 0.0 && sigmaY >= 0.0)
    if (sigmaX == 0 || sigmaY == 0) return 0.0
    val rho = covarianceScores.sum / (sigmaX * sigmaY)
    // If the calculated rho is too large or too small, then the calculation is noisy and we set rho to 0.0
    // TODO (PLA-8597): figure out a better way to rectify covariance noise
    if (rho < -0.999 || rho > 0.999) {
      logger.warn("The covariance estimate is noisy; rectifying. Please consider increasing the ensemble size.")
      0.0
    } else {
      rho
    }
  }
}
