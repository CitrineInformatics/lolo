package io.citrine.lolo.bags

import breeze.stats.distributions.Poisson
import io.citrine.lolo.bags.Bagger.BaggedEnsemble
import io.citrine.lolo.{Learner, Model, TrainingRow}
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
    * @param rng          random number generator for reproducibility
    * @return a training result containing the bagged model
    */
  override def train(trainingData: Seq[TrainingRow[T]], rng: Random = Random()): BaggedTrainingResult[T]

  /** Bootstrap the training data to train an ensemble of models from the base learner. */
  protected def trainEnsemble(trainingData: Seq[TrainingRow[T]], rng: Random): BaggedEnsemble[T] = {
    // Make sure the training data is the same size
    assert(trainingData.forall(trainingData.head.inputs.length == _.inputs.length))
    require(
      trainingData.length >= Bagger.minimumTrainingSize,
      s"We need to have at least ${Bagger.minimumTrainingSize} rows, only ${trainingData.length} given."
    )

    // Set default number of bags
    val actualBags = if (numBags > 0) numBags else trainingData.length

    // We need enough bags such that the probability that the poisson draw is "valid" is at least 50%
    // Valid here means that for each training point, there is at least one tree that doesn't include it
    // The probability that the weights are valid is:
    // (1 - [(1 - 1/e)^{number of trees}])^{number of training points}
    val minBags = math.log(1 - math.pow(2, -1.0 / trainingData.length)) / math.log((Math.E - 1) / math.E)
    require(
      !useJackknife || actualBags >= minBags,
      s"Jackknife requires $minBags bags for ${trainingData.length} training rows, but only $actualBags given."
    )

    // Compute the number of instances of each training row in each training sample
    val Nib = drawNib(actualBags, trainingData.length, rng)

    // Learn the actual models in parallel
    val indices = Nib.indices.toVector
    val (models, importances) = rng
      .zip(indices)
      .par
      .map {
        case (thisRng, i) =>
          val weightedTrainingData = Nib(i).zip(trainingData).map {
            case (count, row) => row.mapWeight(_ * count.toDouble)
          }
          val meta = baseLearner.train(weightedTrainingData, thisRng)
          (meta.getModel(), meta.getFeatureImportance())
      }
      .unzip

    // Average the feature importance
    val averageImportance = importances.reduce(Bagger.combineImportance).map(_.map(_ / importances.size))

    BaggedEnsemble(models, Nib, averageImportance)
  }

  /**
    * Compute the number of instances of each training row in each bag.
    *
    * @return the (# bags) x (# training rows) array of sample counts
    */
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

  override def train(trainingData: Seq[TrainingRow[Double]], rng: Random): RegressionBaggerTrainingResult = {
    // Train the ensemble of models from the data
    val ensemble = trainEnsemble(trainingData, rng)

    // Compute uncertainty rescales and train the bias model (if present)
    val helper = BaggerHelper(ensemble.models, trainingData, ensemble.Nib, useJackknife, uncertaintyCalibration)
    val biasModel = biasLearner.collect {
      case learner if helper.oobErrors.nonEmpty =>
        learner.train(helper.biasTraining, rng = rng).getModel()
    }

    RegressionBaggerTrainingResult(
      ensembleModels = ensemble.models,
      Nib = ensemble.Nib,
      trainingData = trainingData,
      featureImportance = ensemble.averageImportance,
      biasModel = biasModel,
      rescaleRatio = helper.rescaleRatio,
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
  * @tparam T the type of label data
  */
case class ClassificationBagger[T](
    baseLearner: Learner[T],
    numBags: Int = -1,
    useJackknife: Boolean = true,
    disableBootstrap: Boolean = false
) extends Bagger[T] {

  override def train(trainingData: Seq[TrainingRow[T]], rng: Random): ClassificationBaggerTrainingResult[T] = {
    // Train the ensemble of models from the data
    val ensemble = trainEnsemble(trainingData, rng)

    ClassificationBaggerTrainingResult(
      ensembleModels = ensemble.models,
      Nib = ensemble.Nib,
      trainingData = trainingData,
      featureImportance = ensemble.averageImportance,
      disableBootstrap = disableBootstrap
    )
  }
}

object Bagger {

  /**
    * Data class storing intermediate results from training a [[Bagger]]
    *
    * @param models trained from each bag in the ensemble
    * @param Nib array of sample counts of shape (# bags) x (# training rows)
    * @param averageImportance across models in the ensemble
    * @tparam T type of label data for the models
    */
  protected[bags] case class BaggedEnsemble[+T](
      models: ParVector[Model[T]],
      Nib: Vector[Vector[Int]],
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

  /** Combine two optional feature importance vectors. */
  def combineImportance(vec1: Option[Vector[Double]], vec2: Option[Vector[Double]]): Option[Vector[Double]] = {
    vec1.zip(vec2).map {
      case (v1, v2) => v1.zip(v2).map(p => p._1 + p._2)
    }
  }
}
