package io.citrine.lolo.bags

import breeze.linalg.{norm, DenseMatrix, DenseVector}
import breeze.stats.distributions.Poisson
import io.citrine.lolo.stats.StatsUtils
import io.citrine.lolo.{Learner, Model, PredictionResult, RegressionResult, TrainingResult}
import io.citrine.lolo.stats.metrics.RegressionMetrics
import io.citrine.random.Random

import scala.collection.parallel.CollectionConverters._
import scala.collection.parallel.immutable.ParSeq

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data.
  *
  * @param method learner to train each model in the ensemble
  * @param numBags number of base models to aggregate (default of -1 sets the number of models to the number of training rows)
  * @param useJackknife whether to enable jackknife uncertainty estimate
  * @param uncertaintyCalibration whether to enable empirical uncertainty calibration
  * @param disableBootstrap whether to disable bootstrap (useful when `method` implements its own randomization)
  * @param biasLearner learner to use for estimating bias
  */
case class RegressionBagger(
    method: Learner[Double],
    numBags: Int = -1,
    useJackknife: Boolean = true,
    uncertaintyCalibration: Boolean = true,
    disableBootstrap: Boolean = false,
    biasLearner: Option[Learner[Double]] = None
) extends Learner[Double] {
  require(
    !(uncertaintyCalibration && disableBootstrap),
    "Options uncertaintyCalibration and disableBootstrap are incompatible. At most one may be set true."
  )

  /**
    * Draw with replacement from the training data for each model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Double)],
      weights: Option[Seq[Double]],
      rng: Random
  ): RegressionBaggerTrainingResult = {
    /* Make sure the training data is the same size */
    assert(trainingData.forall(trainingData.head._1.size == _._1.size))
    require(
      trainingData.size >= Bagger.minimumTrainingSize,
      s"We need to have at least ${Bagger.minimumTrainingSize} rows, only ${trainingData.size} given"
    )

    /* Use unit weights if none are specified */
    val weightsActual = weights.getOrElse(Seq.fill(trainingData.size)(1.0))

    /* Set default number of bags */
    val actualBags = if (numBags > 0) {
      numBags
    } else {
      trainingData.size
    }

    // We need enough bags such that the probability that the poisson draw is "valid" is at least 50%
    // Valid here means that for each training point, there is at least one tree that doesn't include it
    // The probability that the weights are valid is:
    // (1 - [(1 - 1/e)^{number of trees}])^{number of training points}
    val minBags = Math.log(1 - Math.pow(2, -1.0 / trainingData.size)) / Math.log((Math.E - 1) / Math.E)
    require(
      !useJackknife || actualBags >= minBags,
      s"Jackknife requires $minBags bags for ${trainingData.size} training rows, but only $actualBags given."
    )

    /* Compute the number of instances of each training row in each training sample */
    val randBasis = StatsUtils.breezeRandBasis(rng)
    val dist = new Poisson(1.0)(randBasis)
    val Nib: Vector[Vector[Int]] = if (disableBootstrap) {
      Vector.fill[Vector[Int]](actualBags)(Vector.fill[Int](trainingData.size)(1))
    } else {
      Iterator
        .continually {
          // Generate Poisson distributed weights, filtering out any that don't have the minimum required number
          // of non-zero training weights
          Iterator
            .continually {
              Vector.fill(trainingData.size)(dist.draw())
            }
            .filter(_.count(_ > 0) >= Bagger.minimumNonzeroWeightSize)
            .take(actualBags)
            .toVector
        }
        .filter { nMat =>
          lazy val noAlwaysPresentTrainingData = nMat.transpose.forall { vec => vec.contains(0) }
          // Make sure that at least one learner is missing each training point
          // This prevents a divide-by-zero error in the jackknife-after-bootstrap calculation
          !useJackknife || noAlwaysPresentTrainingData
        }
        .next()
    }

    // Learn the actual models in parallel
    val indices = Nib.indices.toVector
    val (models, importances) = rng
      .zip(indices)
      .par
      .map {
        case (thisRng, i) =>
          val sampleWeights = Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)
          val meta = method.train(trainingData.toVector, Some(sampleWeights), thisRng)
          (meta.getModel(), meta.getFeatureImportance())
      }
      .unzip

    // Average the feature importances
    val averageImportance: Option[Vector[Double]] = importances
      .reduce(Bagger.combineImportance)
      .map(_.map(_ / importances.size))

    // Wrap the models in a BaggedModel object
    val helper = BaggerHelper(models, trainingData, Nib, useJackknife, uncertaintyCalibration)
    val biasModel = biasLearner.collect {
      case learner if helper.oobErrors.nonEmpty =>
        learner.train(helper.biasTraining, rng = rng).getModel()
    }

    new RegressionBaggerTrainingResult(
      models = models,
      Nib = Nib,
      trainingData = trainingData,
      featureImportance = averageImportance,
      biasModel = biasModel,
      rescale = helper.rescaleRatio,
      disableBootstrap = disableBootstrap
    )
  }
}

class RegressionBaggerTrainingResult(
    models: ParSeq[Model[Double]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[(Vector[Any], Double)],
    featureImportance: Option[Vector[Double]],
    biasModel: Option[Model[Double]] = None,
    rescale: Double = 1.0,
    disableBootstrap: Boolean = false
) extends TrainingResult[Double] {

  lazy val NibT: Seq[Vector[Int]] = Nib.transpose
  lazy val model = new BaggedRegressionModel(models, Nib, rescale, disableBootstrap, biasModel)
  lazy val predictedVsActual: Seq[(Vector[Any], Double, Double)] = trainingData.zip(NibT).flatMap {
    case ((f, l), nb) =>
      val oob = if (disableBootstrap) {
        models.zip(nb)
      } else {
        models.zip(nb).filter(_._2 == 0)
      }

      if (oob.isEmpty || l.isNaN) {
        Seq.empty
      } else {
        val predicted = oob.map(_._1.transform(Seq(f)).getExpected().head).sum / oob.size
        Seq((f, predicted, l))
      }
  }

  lazy val loss: Double = RegressionMetrics.RMSE(predictedVsActual)

  override def getFeatureImportance(): Option[Vector[Double]] = featureImportance

  override def getModel(): BaggedRegressionModel = model

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Double, Double)]] = Some(predictedVsActual)

  override def getLoss(): Option[Double] = {
    if (predictedVsActual.nonEmpty) {
      Some(loss)
    } else {
      None
    }
  }
}

/**
  * Container holding a parallel sequence of models and the sample counts used to train them
  *
  * @param models in this bagged model
  * @param Nib    training sample counts
  */
class BaggedRegressionModel(
    val models: ParSeq[Model[Double]],
    Nib: Vector[Vector[Int]],
    rescale: Double = 1.0,
    disableBootstrap: Boolean = false,
    biasModel: Option[Model[Double]] = None
) extends BaggedModel[Double] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult[Double] = {
    assert(inputs.forall(_.size == inputs.head.size))

    val bias = biasModel.map(_.transform(inputs).getExpected())
    val ensemblePredictions = models.map(model => model.transform(inputs)).seq

    if (inputs.size == 1) {
      // In the special case of a single prediction on a real value, emit an optimized prediction class
      SinglePredictionBaggedResult(
        ensemblePredictions,
        Nib,
        bias.map(_.head),
        rescale,
        disableBootstrap
      )
    } else {
      MultiPredictionBaggedResult(
        ensemblePredictions,
        Nib,
        bias,
        rescale,
        disableBootstrap
      )
    }
  }

  /**
    * Compute Shapley feature attributions for a given input
    *
    * @param input for which to compute feature attributions.
    * @param omitFeatures feature indices to omit in computing Shapley values
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  override def shapley(input: Vector[Any], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
    val ensembleShapley = models.map(model => model.shapley(input, omitFeatures))
    if (ensembleShapley.head.isEmpty) {
      None
    } else {
      assert(ensembleShapley.forall(x => x.isDefined))

      def sumReducer(a: Option[DenseMatrix[Double]], b: Option[DenseMatrix[Double]]): Option[DenseMatrix[Double]] = {
        Some(a.get + b.get)
      }
      val scale = 1.0 / ensembleShapley.length

      Some(scale * ensembleShapley.reduce(sumReducer).get)
    }
  }
}

/**
  * Container with model-wise predictions at a single input point.
  *
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
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false
) extends BaggedResult[Double]
    with RegressionResult {

  override def numPredictions: Int = 1

  override def getExpected(): Seq[Double] = Seq(expected + bias.getOrElse(0.0))

  override def getStdDevMean(): Option[Seq[Double]] = {
    if (disableBootstrap) {
      // If bootstrap is disabled, rescale is unity and treeVariance is our only option for UQ.
      // Since it's not recalibrated, it's best considered to be a confidence interval of the underlying weak learner.
      assert(rescaleRatio == 1.0)
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
    treePredictions.map(x => Math.pow(x - expected, 2.0)).sum / (treePredictions.length - 1)
  }

  private lazy val stdDevMean: Double = Math.sqrt(BaggedResult.rectifyEstimatedVariance(singleScores))

  private lazy val stdDevObs: Double = {
    rescaleRatio * Math.sqrt(treeVariance)
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
      val varIJ = Math.pow(cov / vecN.length, 2.0)

      if (tNotCount > 0) {
        // Compute the Jackknife after bootstrap estimate
        val varJ = Math.pow(tNot / tNotCount - expected, 2.0) * (nMat.size - 1) / nMat.size
        // Averaged the corrected IJ and J estimates
        0.5 * (varJ + varIJ - Math.E * correction)
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
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false
) extends BaggedResult[Double]
    with RegressionResult {

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
      assert(rescaleRatio == 1.0)
      Some(varObs.map { v => Math.sqrt(v) })
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

  lazy val NibJMat: DenseMatrix[Double] = BaggedResult.getJackknifeAfterBootstrapMatrix(Nib)

  lazy val NibIJMat: DenseMatrix[Double] = BaggedResult.getInfinitesimalJackknifeMatrix(Nib)

  /* This represents the variance of the estimate of the mean. */
  lazy val stdDevMean: Seq[Double] =
    variance(expected.toVector, expectedMatrix, NibJMat, NibIJMat).map { Math.sqrt }

  /* This estimates the variance of predictive distribution. */
  lazy val varObs: Seq[Double] = expectedMatrix.zip(expected).map {
    case (b, y) =>
      assert(Nib.size > 1, "Bootstrap variance undefined for fewer than 2 bootstrap samples.")
      b.map { x => rescaleRatio * rescaleRatio * Math.pow(x - y, 2.0) }.sum / (b.size - 1)
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
    scores(meanPrediction, modelPredictions, NibJ, NibIJ).map { BaggedResult.rectifyEstimatedVariance }
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
    val prefactor = 1.0 / Math.pow(modelPredictions.head.size, 2.0) * (Nib.size - 1) / Nib.size

    modelPredictions.indices
      .map { i =>
        /* Compute the first order bias correction for the variance estimators */
        val correction = prefactor * Math.pow(norm(predMat(::, i) - meanPrediction(i)), 2)

        /* The correction is prediction dependent, so we need to operate on vectors */
        0.5 * (arg(::, i) - Math.E * correction)
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
          Math.signum(actualPrediction(i) - meanPrediction(i)) * 0.5 * (arg(::, i) - Math.E * correction)

        /* Impose a floor in case any of the variances are negative (hacked to work in breeze) */
        // val floor: Double = Math.min(0, -min(variancePerRow))
        // val rezero: DenseVector[Double] = variancePerRow - floor
        // 0.5 * (rezero + abs(rezero)) + floor
        influencePerRow
      }
      .map(_.toScalaVector)
  }
}
