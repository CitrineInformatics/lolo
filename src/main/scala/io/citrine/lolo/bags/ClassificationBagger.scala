package io.citrine.lolo.bags

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Poisson
import io.citrine.lolo.stats.StatsUtils
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.random.Random

import scala.collection.parallel.CollectionConverters._
import scala.collection.parallel.immutable.ParSeq

/**
  * A bagger creates an ensemble of models by training the learner on random samples of the training data
  *
  * @param method learner to train each model in the ensemble
  * @param numBags number of base models to aggregate (default of -1 sets the number of models to the number of training rows)
  * @param useJackknife whether to enable jackknife uncertainty estimate
  * @param disableBootstrap whether to disable bootstrap (useful when `method` implements its own randomization)
  */
case class ClassificationBagger(
    method: Learner[Any],
    numBags: Int = -1,
    useJackknife: Boolean = true,
    disableBootstrap: Boolean = false
) extends Learner[Any] {

  /**
    * Draw with replacement from the training data for each model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): ClassificationBaggerTrainingResult = {
    /* Make sure the training data is the same size */
    assert(trainingData.forall(trainingData.head._1.size == _._1.size))

    require(
      trainingData.size >= Bagger.minimumTrainingSize,
      s"We need to have at least ${Bagger.minimumTrainingSize} rows, only ${trainingData.size} given"
    )

    /* Use unit weights if none are specified */
    val weightsActual = weights.getOrElse(Seq.fill(trainingData.size)(1.0))

    /* Set default number of bags */
    val actualBags = if (numBags > 0) numBags else trainingData.size

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

    new ClassificationBaggerTrainingResult(
      models = models,
      Nib = Nib,
      trainingData = trainingData,
      featureImportance = averageImportance,
      disableBootstrap = disableBootstrap
    )
  }
}

class ClassificationBaggerTrainingResult(
    models: ParSeq[Model[Any]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[(Vector[Any], Any)],
    featureImportance: Option[Vector[Double]],
    disableBootstrap: Boolean = false
) extends TrainingResult[Any] {

  lazy val NibT: Seq[Vector[Int]] = Nib.transpose
  lazy val model = new BaggedClassificationModel(models, Nib, disableBootstrap)
  lazy val predictedVsActual: Seq[(Vector[Any], Any, Any)] = trainingData.zip(NibT).flatMap {
    case ((f, l), nb) =>
      val oob = if (disableBootstrap) {
        models.zip(nb)
      } else {
        models.zip(nb).filter(_._2 == 0)
      }

      if (oob.isEmpty || l == null) {
        Seq()
      } else {
        val predicted = oob.map(_._1.transform(Seq(f)).getExpected().head).groupBy(identity).maxBy(_._2.size)._1
        Seq((f, predicted, l))
      }
  }

  lazy val loss: Double = ClassificationMetrics.loss(predictedVsActual)

  override def getFeatureImportance(): Option[Vector[Double]] = featureImportance

  override def getModel(): BaggedClassificationModel = model

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Any, Any)]] = Some(predictedVsActual)

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
class BaggedClassificationModel(
    val models: ParSeq[Model[Any]],
    Nib: Vector[Vector[Int]],
    disableBootstrap: Boolean = false
) extends BaggedModel[Any] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedClassificationResult = {
    assert(inputs.forall(_.size == inputs.head.size))
    val ensemblePredictions = models.map(model => model.transform(inputs)).seq
    BaggedClassificationResult(ensemblePredictions)
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

case class BaggedClassificationResult(predictions: Seq[PredictionResult[Any]]) extends BaggedResult[Any] {
  lazy val expectedMatrix: Seq[Seq[Any]] = predictions.map(p => p.getExpected()).transpose
  lazy val expected: Seq[Any] = expectedMatrix.map(ps => ps.groupBy(identity).maxBy(_._2.size)._1)
  lazy val uncertainty: Seq[Map[Any, Double]] =
    expectedMatrix.map(ps => ps.groupBy(identity).view.mapValues(_.size.toDouble / ps.size).toMap)

  override def numPredictions: Int = expectedMatrix.length

  override def getExpected(): Seq[Any] = expected

  override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Map[Any, Double]]] = Some(uncertainty)
}
