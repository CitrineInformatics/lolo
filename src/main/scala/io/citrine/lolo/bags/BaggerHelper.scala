package io.citrine.lolo.bags

import io.citrine.lolo.stats.{MathUtils, StatsUtils}
import io.citrine.lolo.{Model, PredictionResult, RegressionResult}

import scala.collection.parallel.immutable.ParSeq

/**
  * Helper class to subsume shared functionality of Bagger and MultiTaskBagger.
  *
  * @param models collection of trained models
  * @param trainingData on which models were trained
  * @param Nib vector (over models) of vectors (over training data) of the number of repeats in each model's bag
  * @param useJackknife whether to use jackknife for uncertainty quantification
  * @param uncertaintyCalibration whether to apply empirical uncertainty calibration
  */
protected case class BaggerHelper(
    models: ParSeq[Model[Any]],
    trainingData: Seq[(Vector[Any], Any)],
    Nib: Vector[Vector[Int]],
    useJackknife: Boolean,
    uncertaintyCalibration: Boolean
) {
  val isRegression: Boolean = trainingData.map { _._2 }.find { _ != null } match {
    case Some(_: Double) => true
    case Some(_: Any)    => false
    case None            => throw new IllegalArgumentException(s"Unable to find a non-null label")
  }

  /**
    * Seq of tuples containing training data paired with their out-of-bag residuals and uncertainty estimates.
    * Defined as lazy so it's only computed if needed for the ratio or bias learner calculations.
    */
  lazy val oobErrors: Seq[(Vector[Any], Double, Double)] = trainingData.indices.flatMap { idx =>
    val oobModels = models.zip(Nib.map(_(idx))).filter(_._2 == 0).map(_._1).asInstanceOf[ParSeq[Model[Double]]]
    val label = trainingData(idx)._2
    if (oobModels.size < 2 || label == null || (label.isInstanceOf[Double] && label.asInstanceOf[Double].isNaN)) {
      None
    } else {
      val model = new BaggedModel(
        oobModels,
        Nib.filter { _(idx) == 0 },
        useJackknife
      )
      val predicted = model.transform(Seq(trainingData(idx)._1))
      val error = predicted.getExpected().head - trainingData(idx)._2.asInstanceOf[Double]
      val uncertainty = predicted match {
        case x: RegressionResult => x.getStdDevObs().get.head
        case _: Any =>
          throw new UnsupportedOperationException("Computing oobErrors for classification is not supported.")
      }
      Some(trainingData(idx)._1, error, uncertainty)
    }
  }

  /** Uncertainty calibration ratio based on the OOB errors. */
  val rescaleRatio: Double = if (uncertaintyCalibration && isRegression && useJackknife) {
    val trainingLabels = trainingData.collect { case (_, x: Double) if !(x.isInfinite || x.isNaN) => x }
    val zeroTolerance = StatsUtils.range(trainingLabels) / 1e12
    BaggerHelper.calculateRescaleRatio(oobErrors.map { case (_, e, u) => (e, u) }, zeroTolerance = zeroTolerance)
  } else {
    1.0
  }
  assert(!rescaleRatio.isNaN && !rescaleRatio.isInfinity, s"Uncertainty calibration ratio is not real: $rescaleRatio")

  /**
    * Data on which to train a bias learner.
    */
  lazy val biasTraining: Seq[(Vector[Any], Double)] = oobErrors.map {
    case (f, e, u) =>
      // Math.E is only statistically correct.  It should be actualBags / Nib.transpose(i).count(_ == 0)
      // Or, better yet, filter the bags that don't include the training example
      val bias = Math.E * Math.max(Math.abs(e) - u * rescaleRatio, 0)
      (f, bias)
  }
}

object BaggerHelper {

  /**
    * Calculate the uncertainty calibration ratio, which is the 68th percentile of error/uncertainty for the training points.
    *
    * If a point has 0 uncertainty, the ratio is 1 iff error is also 0, or infinity otherwise.
    * If the 68th percentile ratio is infinity, default to 1.0.
    * This is not unreasonable when the number of training data and bags are small,
    * meaning there may only be 2 out-of-bag models.
    *
    * @param oobErrors sequence of tuples of (error, uncertainty) from OOB predictions for each training row
    * @param zeroTolerance tolerance for determining if the error/uncertainty are zero
    * @return uncertainty rescale ratio
    */
  def calculateRescaleRatio(oobErrors: Seq[(Double, Double)], zeroTolerance: Double = 1e-12): Double = {
    val oneSigmaRatio = oobErrors
      .map {
        case (error, uncertainty) =>
          val errorIsZero = MathUtils.tolerantEquals(error, 0.0, zeroTolerance)
          val uncertaintyIsZero = MathUtils.tolerantEquals(uncertainty, 0.0, zeroTolerance)
          if (errorIsZero && uncertaintyIsZero) {
            1.0
          } else {
            math.abs(error / uncertainty)
          }
      }
      .sorted
      .drop((oobErrors.size * 0.68).toInt)
      .headOption
      .getOrElse(1.0)
    if (oneSigmaRatio.isPosInfinity) {
      1.0
    } else {
      oneSigmaRatio
    }
  }
}
