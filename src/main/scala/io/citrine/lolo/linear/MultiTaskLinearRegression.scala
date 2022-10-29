package io.citrine.lolo.linear

import io.citrine.lolo.api.{MultiTaskLearner, MultiTaskModel, MultiTaskTrainingResult, ParallelModels, TrainingRow}
import io.citrine.random.Random

case class MultiTaskLinearRegressionLearner(regParam: Option[Double] = None, fitIntercept: Boolean = true)
    extends MultiTaskLearner {

  override def train(trainingData: Seq[TrainingRow[Vector[Any]]], rng: Random): MultiTaskTrainingResult = {
    val rep = trainingData.head
    val repLabels = rep.label
    val numLabels = repLabels.length
    assert(repLabels.forall(_.isInstanceOf[Double]))

    val singleTaskLearner = LinearRegressionLearner(regParam, fitIntercept)
    val singleTaskRows = Vector.tabulate(numLabels) { idx =>
      trainingData.map(_.mapLabel(labelVec => labelVec(idx).asInstanceOf[Double]))
    }
    val singleTaskResults = singleTaskRows.map { labelTrainingRows =>
      singleTaskLearner.train(labelTrainingRows, rng)
    }
    MultiTaskLinearRegressionTrainingResult(singleTaskResults)
  }
}

case class MultiTaskLinearRegressionTrainingResult(linearResults: Vector[LinearRegressionTrainingResult])
    extends MultiTaskTrainingResult {

  override def models: Vector[LinearRegressionModel] = linearResults.map(_.model)

  override def model: MultiTaskModel = ParallelModels(models, Vector.fill(models.length)(true))
}
