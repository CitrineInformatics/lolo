package io.citrine.lolo.bags

import io.citrine.lolo.api.{Model, TrainingResult, TrainingRow}
import io.citrine.lolo.stats.metrics.{ClassificationMetrics, RegressionMetrics}

import scala.collection.parallel.immutable.ParSeq

/** The result of training a [[Bagger]] to produce a [[BaggedModel]]. */
sealed trait BaggedTrainingResult[+T] extends TrainingResult[T] {

  override def model: BaggedModel[T]
}

case class RegressionBaggerTrainingResult(
    ensembleModels: ParSeq[Model[Double]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[TrainingRow[Double]],
    override val featureImportance: Option[Vector[Double]],
    biasModel: Option[Model[Double]] = None,
    rescaleRatio: Double = 1.0,
    disableBootstrap: Boolean = false
) extends BaggedTrainingResult[Double] {

  lazy val NibT: Vector[Vector[Int]] = Nib.transpose

  override lazy val model: BaggedRegressionModel =
    BaggedRegressionModel(ensembleModels, Nib, rescaleRatio, disableBootstrap, biasModel)

  override lazy val predictedVsActual: Option[Seq[(Vector[Any], Double, Double)]] = Some(
    trainingData.zip(NibT).flatMap {
      case (TrainingRow(f, l, _), nb) =>
        val oob = if (disableBootstrap) {
          ensembleModels.zip(nb)
        } else {
          ensembleModels.zip(nb).filter(_._2 == 0)
        }

        if (oob.isEmpty || l.isNaN) {
          Seq.empty
        } else {
          val predicted = oob.map(_._1.transform(Seq(f)).expected.head).sum / oob.size
          Seq((f, predicted, l))
        }
    }
  )

  override lazy val loss: Option[Double] = predictedVsActual.collect {
    case pva if pva.nonEmpty => RegressionMetrics.RMSE(pva)
  }
}

case class ClassificationBaggerTrainingResult[T](
    ensembleModels: ParSeq[Model[T]],
    Nib: Vector[Vector[Int]],
    trainingData: Seq[TrainingRow[T]],
    override val featureImportance: Option[Vector[Double]],
    disableBootstrap: Boolean = false
) extends BaggedTrainingResult[T] {

  lazy val NibT: Vector[Vector[Int]] = Nib.transpose

  override lazy val model: BaggedClassificationModel[T] = BaggedClassificationModel(ensembleModels)

  override lazy val predictedVsActual: Option[Seq[(Vector[Any], T, T)]] = Some(
    trainingData.zip(NibT).flatMap {
      case (TrainingRow(f, l, _), nb) =>
        val oob = if (disableBootstrap) {
          ensembleModels.zip(nb)
        } else {
          ensembleModels.zip(nb).filter(_._2 == 0)
        }

        if (oob.isEmpty || l == null) {
          Seq()
        } else {
          val predicted = oob.map(_._1.transform(Seq(f)).expected.head).groupBy(identity).maxBy(_._2.size)._1
          Seq((f, predicted, l))
        }
    }
  )

  override lazy val loss: Option[Double] = predictedVsActual.collect {
    case pva if pva.nonEmpty => ClassificationMetrics.loss(pva)
  }
}
