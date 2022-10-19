package io.citrine.lolo.bags

import io.citrine.lolo.stats.metrics.{ClassificationMetrics, RegressionMetrics}
import io.citrine.lolo.{Model, TrainingResult}

import scala.collection.parallel.immutable.ParSeq

/** The result of training a [[Bagger]] to produce a [[BaggedModel]]. */
sealed trait BaggedTrainingResult[+T] extends TrainingResult[T] {

  override def getModel(): BaggedModel[T]
}

case class RegressionBaggerTrainingResult(
    ensembleModels: ParSeq[Model[Double]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[(Vector[Any], Double)],
    featureImportance: Option[Vector[Double]],
    biasModel: Option[Model[Double]] = None,
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false
) extends BaggedTrainingResult[Double] {

  lazy val NibT: Vector[Vector[Int]] = Nib.transpose
  lazy val model: BaggedRegressionModel =
    BaggedRegressionModel(ensembleModels, Nib, rescaleRatio, disableBootstrap, biasModel)
  lazy val predictedVsActual: Seq[(Vector[Any], Double, Double)] = trainingData.zip(NibT).flatMap {
    case ((f, l), nb) =>
      val oob = if (disableBootstrap) {
        ensembleModels.zip(nb)
      } else {
        ensembleModels.zip(nb).filter(_._2 == 0)
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

case class ClassificationBaggerTrainingResult[T](
    ensembleModels: ParSeq[Model[T]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[(Vector[Any], T)],
    featureImportance: Option[Vector[Double]],
    disableBootstrap: Boolean = false
) extends BaggedTrainingResult[T] {

  lazy val NibT: Vector[Vector[Int]] = Nib.transpose
  lazy val model: BaggedClassificationModel[T] = BaggedClassificationModel(ensembleModels, Nib)
  lazy val predictedVsActual: Seq[(Vector[Any], T, T)] = trainingData.zip(NibT).flatMap {
    case ((f, l), nb) =>
      val oob = if (disableBootstrap) {
        ensembleModels.zip(nb)
      } else {
        ensembleModels.zip(nb).filter(_._2 == 0)
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

  override def getModel(): BaggedClassificationModel[T] = model

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], T, T)]] = Some(predictedVsActual)

  override def getLoss(): Option[Double] = {
    if (predictedVsActual.nonEmpty) {
      Some(loss)
    } else {
      None
    }
  }
}
