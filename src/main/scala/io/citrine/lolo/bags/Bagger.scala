package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo.bags.Bagger.BaggedEnsemble
import io.citrine.lolo.{Learner, Model}
import io.citrine.lolo.stats.StatsUtils
import io.citrine.random.Random

import scala.collection.parallel.CollectionConverters._
import scala.collection.parallel.immutable.ParVector

sealed trait Bagger[T] extends Learner[T] {

  def numBags: Int

  def useJackknife: Boolean

  def disableBootstrap: Boolean

  def baseLearner: Learner[T]

  /**
    * Draw with replacement from the training data for each model.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a training result containing the bagged model
    */
  override def train(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]],
      rng: Random
  ): BaggedTrainingResult[T]

  /** Bootstrap the training data to train an ensemble of models from the base learner. */
  protected def trainEnsemble(
      trainingData: Seq[(Vector[Any], T)],
      weights: Option[Seq[Double]],
      rng: Random
  ): BaggedEnsemble[T] = {
    // Make sure the training data is the same size
    assert(trainingData.forall(trainingData.head._1.length == _._1.length))

    // Check size requirements and determine usable number of bags
    val actualBags = getActualBags(trainingData.length)

    // Use unit weights if none are specified
    val weightsActual = weights.getOrElse(Seq.fill(trainingData.length)(1.0))

    // Compute the number of instances of each training row in each training sample
    val Nib = drawNib(actualBags, trainingData.length, rng)

    // Learn the actual models in parallel
    val indices = Nib.indices.toVector
    val (models, importances) = rng
      .zip(indices)
      .par
      .map {
        case (thisRng, i) =>
          val sampleWeights = Nib(i).zip(weightsActual).map(p => p._1.toDouble * p._2)
          val meta = baseLearner.train(trainingData, Some(sampleWeights), thisRng)
          (meta.getModel(), meta.getFeatureImportance())
      }
      .unzip

    // Average the feature importances
    val averageImportance = importances.reduce(Bagger.combineImportance).map(_.map(_ / importances.size))

    BaggedEnsemble(Nib, models, averageImportance)
  }

  /**
    * Determine the actual number of bags to use for a given training size
    * while asserting size requirements of the ensemble and training data.
    */
  private def getActualBags(trainingSize: Int): Int = {
    require(
      trainingSize >= Bagger.minimumTrainingSize,
      s"We need to have at least ${Bagger.minimumTrainingSize} rows, only $trainingSize given."
    )

    // Set default number of bags
    val actualBags = if (numBags > 0) numBags else trainingSize

    // We need enough bags such that the probability that the poisson draw is "valid" is at least 50%
    // Valid here means that for each training point, there is at least one tree that doesn't include it
    // The probability that the weights are valid is:
    // (1 - [(1 - 1/e)^{number of trees}])^{number of training points}
    val minBags = math.log(1 - math.pow(2, -1.0 / trainingSize)) / math.log((Math.E - 1) / math.E)
    require(
      !useJackknife || actualBags >= minBags,
      s"Jackknife requires $minBags bags for $trainingSize training rows, but only $actualBags given."
    )

    actualBags
  }

  /** Compute the number of instances of each training row in each training sample. */
  private def drawNib(actualBags: Int, trainingSize: Int, rng: Random = Random()): Vector[Vector[Int]] = {
    val randBasis = StatsUtils.breezeRandBasis(rng)
    val dist = new Poisson(1.0)(randBasis)
    if (disableBootstrap) {
      Vector.fill[Vector[Int]](actualBags)(Vector.fill[Int](trainingSize)(1))
    } else {
      Iterator
        .continually {
          // Generate Poisson distributed weights, filtering out any that don't have the minimum required number
          // of non-zero training weights
          Iterator
            .continually {
              Vector.fill(trainingSize)(dist.draw())
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
  }
}

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data.
  *
  * @param baseLearner learner to train each model in the ensemble
  * @param numBags number of base models to aggregate (default of -1 sets the number of models to the number of training rows)
  * @param useJackknife whether to enable jackknife uncertainty estimate
  * @param uncertaintyCalibration whether to enable empirical uncertainty calibration
  * @param disableBootstrap whether to disable bootstrap (useful when `method` implements its own randomization)
  * @param biasLearner learner to use for estimating bias
  */
case class RegressionBagger(
    baseLearner: Learner[Double],
    numBags: Int = -1,
    useJackknife: Boolean = true,
    uncertaintyCalibration: Boolean = true,
    disableBootstrap: Boolean = false,
    biasLearner: Option[Learner[Double]] = None
) extends Bagger[Double] {
  require(
    !(uncertaintyCalibration && disableBootstrap),
    "Options uncertaintyCalibration and disableBootstrap are incompatible. At most one may be set true."
  )

  override def train(
      trainingData: Seq[(Vector[Any], Double)],
      weights: Option[Seq[Double]],
      rng: Random
  ): RegressionBaggerTrainingResult = {
    // Train the ensemble of models from the data
    val ensemble = trainEnsemble(trainingData, weights, rng)

    // Compute uncertainty rescales and train the bias model (if present)
    val helper = BaggerHelper(ensemble.models, trainingData, ensemble.Nib, useJackknife, uncertaintyCalibration)
    val biasModel = biasLearner.collect {
      case learner if helper.oobErrors.nonEmpty =>
        learner.train(helper.biasTraining, rng = rng).getModel()
    }

    new RegressionBaggerTrainingResult(
      models = ensemble.models,
      Nib = ensemble.Nib,
      trainingData = trainingData,
      featureImportance = ensemble.averageImportance,
      biasModel = biasModel,
      rescale = helper.rescaleRatio,
      disableBootstrap = disableBootstrap
    )
  }
}

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data
  *
  * @param baseLearner learner to train each model in the ensemble
  * @param numBags number of base models to aggregate (default of -1 sets the number of models to the number of training rows)
  * @param useJackknife whether to enable jackknife uncertainty estimate
  * @param disableBootstrap whether to disable bootstrap (useful when `method` implements its own randomization)
  */
case class ClassificationBagger(
    baseLearner: Learner[Any],
    numBags: Int = -1,
    useJackknife: Boolean = true,
    disableBootstrap: Boolean = false
) extends Bagger[Any] {

  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): ClassificationBaggerTrainingResult = {
    // Train the ensemble of models from the data
    val ensemble = trainEnsemble(trainingData, weights, rng)

    new ClassificationBaggerTrainingResult(
      models = ensemble.models,
      Nib = ensemble.Nib,
      trainingData = trainingData,
      featureImportance = ensemble.averageImportance,
      disableBootstrap = disableBootstrap
    )
  }
}

object Bagger {

  /** Ensemble of models derived from training a [[Bagger]]. */
  case class BaggedEnsemble[T](
      Nib: Vector[Vector[Int]],
      models: ParVector[Model[T]],
      averageImportance: Option[Vector[Double]]
  )

  /**
    * The minimum number of training rows in order to train a Bagger
    */
  val minimumTrainingSize: Int = 8

  /** The minimum number of training examples with a value for any given output */
  val minimumOutputCount: Int = 2

  /**
    * The minimum number of non-zero weighted training points that is based into the learner
    *
    * This requirement biases the poisson draws, but hopefully not too too much
    */
  val minimumNonzeroWeightSize: Int = 4

  def combineImportance(v1: Option[Vector[Double]], v2: Option[Vector[Double]]): Option[Vector[Double]] = {
    (v1, v2) match {
      case (None, None)                                         => None
      case (Some(v1: Vector[Double]), Some(v2: Vector[Double])) => Some(v1.zip(v2).map(p => p._1 + p._2))
      case _                                                    => None
    }
  }
}
