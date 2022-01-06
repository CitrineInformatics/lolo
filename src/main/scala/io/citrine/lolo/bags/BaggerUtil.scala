package io.citrine.lolo.bags

import io.citrine.lolo.util.Async
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
                                   models: ParSeq[Model[PredictionResult[Any]]],
                                   trainingData: Seq[(Vector[Any],Any)],
                                   Nib: Vector[Vector[Int]],
                                   useJackknife: Boolean,
                                   uncertaintyCalibration: Boolean
                                 ) {

  /** Does the training data represent a regression problem? */
  val isRegression: Boolean = trainingData.map{_._2}.find{ _ != null } match {
    case Some(_: Double) => true
    case Some(_: Any) => false
    case None => throw new IllegalArgumentException(s"Unable to find a non-null label")
  }

  /**
    * Seq of tuples containing training data paired with their out-of-bag residuals and uncertainty estimates.
    */
  lazy val oobErrors: Seq[(Vector[Any], Double, Double)] = trainingData.indices.flatMap { idx =>
    val oobModels = models.zip(Nib.map(_ (idx))).filter(_._2 == 0).map(_._1).asInstanceOf[ParSeq[Model[PredictionResult[Double]]]]
    val label = trainingData(idx)._2
    if (oobModels.size < 2 || label == null || (label.isInstanceOf[Double] && label.asInstanceOf[Double].isNaN)) {
      None
    } else {
      Async.canStop()
      val model = new BaggedModel(oobModels, Nib.filter {
        _ (idx) == 0
      }, useJackknife)
      val predicted = model.transform(Seq(trainingData(idx)._1))
      val error = predicted.getExpected().head - trainingData(idx)._2.asInstanceOf[Double]
      val uncertainty = predicted match {
        case x: RegressionResult => x.getStdDevObs.get.head
        case _: Any => throw new UnsupportedOperationException("Computing oobErrors for classification is not supported.")
      }
      Some(trainingData(idx)._1, error, uncertainty)
    }
  }

  /**
    * Calculate the uncertainty calibration ratio, which is the 68th percentile of error/uncertainty.
    * for the training points. If a point has 0 uncertainty, the ratio is 1 iff error is also 0, or infinity otherwise.
    */
  val ratio: Double = if (uncertaintyCalibration && isRegression && useJackknife) {
    Async.canStop()
    oobErrors.map {
      case (_, 0.0, 0.0) => 1.0
      case (_, _, 0.0) => Double.PositiveInfinity
      case (_, error, uncertainty) => Math.abs(error / uncertainty)
    }.sorted.drop((oobErrors.size * 0.68).toInt).head
  } else {
    1.0
  }
  assert(!ratio.isNaN && !ratio.isInfinity, s"Uncertainty calibration ratio is not real: $ratio")

  /**
    * Data on which to train a bias learner.
    */
  lazy val biasTraining: Seq[(Vector[Any], Double)] = oobErrors.map { case (f, e, u) =>
    // Math.E is only statistically correct.  It should be actualBags / Nib.transpose(i).count(_ == 0)
    // Or, better yet, filter the bags that don't include the training example
    val bias = Math.E * Math.max(Math.abs(e) - u * ratio, 0)
    (f, bias)
  }
}
