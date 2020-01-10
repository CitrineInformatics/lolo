package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix, DenseVector, eig, max, min, norm, sum}
import breeze.numerics.abs
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.util.Async
import sun.reflect.generics.reflectiveObjects.NotImplementedException

/**
  * Interface defining the return value of a [[BaggedModel]]
  *
  * This allows the implementation to depend on the number of simultaneous predictions, which has performance
  * implications.
  */
trait BaggedResult extends PredictionResult[Any] {
  protected val predictions: Seq[PredictionResult[Any]]

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
  * @param repInput    representative input
  */
case class BaggedSingleResult(
                               predictions: Seq[PredictionResult[Any]],
                               NibIn: Vector[Vector[Int]],
                               useJackknife: Boolean,
                               bias: Option[Double],
                               repInput: Vector[Any],
                               rescale: Double = 1.0
                             ) extends BaggedResult {
  assert(predictions.head.getExpected().head.isInstanceOf[Double])
  private lazy val treePredictions = predictions.map(_.getExpected().head.asInstanceOf[Double]).toArray

  /**
    * Return the ensemble average or maximum vote
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = Seq(expected)

  private lazy val expected = treePredictions.sum / treePredictions.size

  /**
    * Return jackknife-based variance estimates
    *
    * @return uncertainty of each prediction
    */
  override def getUncertainty(): Option[Seq[Any]] = Some(Seq(scalarUncertainty))

  private lazy val scalarUncertainty = if (useJackknife) {
    Math.sqrt(singleScores.sum * Math.pow(rescale, 2.0)) //  + Math.pow(bias.getOrElse(0.0), 2.0))
  } else {
    Math.sqrt(ensembleVariance * Math.pow(rescale, 2.0)) //  + Math.pow(bias.getOrElse(0.0), 2.0))
  }

  /**
    * The importances are computed as an average of bias-corrected jackknife-after-bootstrap
    * and infinitesimal jackknife methods
    *
    * @return training row scores of each prediction
    */
  override def getImportanceScores(): Option[Seq[Seq[Double]]] = Some(Seq(singleScores))

  private lazy val singleScores: Vector[Double] = {
    // Compute the variance of the ensemble of predicted values and divide by the size of the ensemble an extra time
    val varT = 1.0 / (treePredictions.size * treePredictions.size) * treePredictions.map(y => Math.pow(y - expected, 2.0)).sum

    // This will be more convenient later
    val nMat = NibIn.transpose

    // Loop over each of the training instances, computing its contribution to the uncertainty
    var minimumContribution: Double = Double.MaxValue
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

      val res = if (tNotCount > 0) {
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

      // Keep track of the smallest estimated uncertainty, which may be negative
      minimumContribution = Math.min(minimumContribution, res)
      res
    }
    // The uncertainty must be positive, so anything smaller than zero is noise.  Make sure that no estimated
    // uncertainty is below that noise level
    val floor = Math.max(0, -minimumContribution)
    trainingContributions.map { x => Math.max(x, floor) }
  }

  lazy val ensembleVariance: Double = {
    treePredictions.map(x => Math.pow(x - expected, 2.0)).sum / (treePredictions.size - 1)
  }
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
  * @param repInput    representative input
  */
case class BaggedMultiResult(
                         override protected val predictions: Seq[PredictionResult[Any]],
                         NibIn: Vector[Vector[Int]],
                         useJackknife: Boolean,
                         bias: Option[Seq[Any]],
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

  def getPredictedVariance(): Double = {
    val actuals = getExpected()
    if (!actuals.head.isInstanceOf[Double]) {
      throw new NotImplementedException()
    }
    val residuals: Seq[Seq[Double]] = expectedMatrix.zip(actuals).map{case (pred, y) =>
      pred.map(_.asInstanceOf[Double] - y.asInstanceOf[Double])
    }.transpose

    val sd = residuals.map{vec =>
      Math.sqrt(vec.map(Math.pow(_, 2.0)).sum / vec.size)
    }

    Seq.tabulate(sd.size, sd.size){ case (i, j) =>
      residuals(i).zip(residuals(j)).map(x => x._1 * x._2).sum / residuals(i).size
    }.flatten.sum / (residuals.size * residuals.size)
  }

  def getPredictiveCorrelation(): Double = {
    val actuals = getExpected()
    if (!actuals.head.isInstanceOf[Double]) {
      throw new NotImplementedException()
    }

    val residuals: Seq[Seq[Double]] = expectedMatrix.zip(actuals).map{case (pred, y) =>
      pred.map(_.asInstanceOf[Double] - y.asInstanceOf[Double])
    }.transpose

    val rho = Seq.tabulate(residuals.size, residuals.size){ case (i, j) =>
        residuals(i).zip(residuals(j)).map(x => x._1 * x._2).sum / residuals(i).size
    }

    val rMat = new DenseMatrix[Double](rho.size, rho.size, rho.flatten.toArray)
    eig(rMat).eigenvalues.toArray.map(Math.pow(_, 2.0)).sum / rho.indices.map(i => Math.pow(rho(i)(i), 2.0)).sum
  }

  def getCorrelation(actuals: Seq[Any]): Double = {
    if (!actuals.head.isInstanceOf[Double]) {
      throw new NotImplementedException()
    }
    val residuals: Seq[Seq[Double]] = expectedMatrix.zip(actuals).map{case (pred, y) =>
        pred.map(_.asInstanceOf[Double] - y.asInstanceOf[Double])
    }.transpose

    val sd = residuals.map{vec =>
      Math.sqrt(vec.map(Math.pow(_, 2.0)).sum / vec.size)
    }
    val avgSd = sd.sum / sd.size

    val rho = Seq.tabulate(sd.size, sd.size){ case (i, j) =>
      if (i == j) {
        0.0
      } else {
        residuals(i).zip(residuals(j)).map(x => x._1 * x._2).sum / residuals(i).size
      }
    }.flatten.sum / Math.pow(avgSd, 2.0) / (sd.size * (sd.size - 1))
    rho
  }

  def getTreeError(actuals: Seq[Any]): Double = {
    if (!actuals.head.isInstanceOf[Double]) {
      throw new NotImplementedException()
    }
    val residuals: Seq[Seq[Double]] = expectedMatrix.zip(actuals).map{case (pred, y) =>
      pred.map(_.asInstanceOf[Double] - y.asInstanceOf[Double])
    }.transpose

    val sd = residuals.map{vec =>
      vec.map(Math.pow(_, 2.0)).sum / vec.size
    }
    sd.sum / sd.size
  }

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
      v.map(n => n * itot - vtot)
    }.toArray
  )

  /* This matrix is used to compute the IJ variance */
  lazy val biasIJMat = new DenseMatrix[Double](Nib.head.size, Nib.size,
    Nib.flatMap { v =>
      val itot = 1.0 / v.size
      val vtot = v.sum.toDouble * itot
      v.map(n => itot * itot * Math.pow(n - vtot, 2.0))
    }.toArray
  )

  def getEnsembleVariance(): Seq[Double] = {
    ensembleVariance.map(_ * rescale * rescale)
  }

  lazy val ensembleVariance: Seq[Double] = expectedMatrix.asInstanceOf[Seq[Seq[Double]]].zip(expected.asInstanceOf[Seq[Double]]).map { case (b, y) =>
    b.map { x => Math.pow(x - y, 2.0) }.sum / (b.size - 1)
  }

  lazy val jackknifeVariance: Seq[Double] = variance(
    expected.asInstanceOf[Seq[Double]].toVector,
    expectedMatrix.asInstanceOf[Seq[Seq[Double]]],
    NibJMat, NibIJMat
  )

  lazy val biasEstimate: Seq[Double] = bias.getOrElse(Seq.fill(expected.size)(0.0)).asInstanceOf[Seq[Double]]

  /* Compute the uncertainties one prediction at a time */
  lazy val uncertainty: Seq[Any] = {
    rep match {
      case x: Double =>
        if (useJackknife) {
          jackknifeVariance.zip(biasEstimate).map { case (variance, b) => Math.sqrt(b * b + variance) * rescale}
        } else {
          ensembleVariance.map(v => Math.sqrt(v) * rescale)
        }
      case x: Any =>
        expectedMatrix.map(ps => ps.groupBy(identity).mapValues(_.size.toDouble / ps.size))
    }
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
    scores(meanPrediction, modelPredictions, NibJ, NibIJ).map{vec =>
      val importances = vec.sorted.reverse
      val cutoff = Double.NegativeInfinity
      val signal = importances.filter(_ > cutoff)
      val res = signal.sum
      if (res <= 0) {
        val test = importances.filter(_ > 0).sum
        if (test <= 0) {
          - importances.last
        } else {
          test
        }
      } else {
        // println(signal.size.toDouble / importances.size)
        res
      }
    }
  } ensuring(_.forall(x => x > 0))

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
      val variancePerRow: DenseVector[Double] = 0.5 * (arg(::, i) - Math.E * correction)

      /* Impose a floor in case any of the variances are negative (hacked to work in breeze) */
      val floor: Double = Math.max(0, -min(variancePerRow))
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


