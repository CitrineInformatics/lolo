package io.citrine.lolo.linear

import io.citrine.lolo.api.{MultiTaskLearner, MultiTaskModel, MultiTaskTrainingResult, ParallelModels, TrainingRow}
import io.citrine.random.Random

case class MultiTaskLinearRegressionLearner(regParam: Option[Double] = None, fitIntercept: Boolean = true)
    extends MultiTaskLearner {

  override def train(
      trainingData: Seq[TrainingRow[Vector[Any]]],
      rng: Random
  ): MultiTaskLinearRegressionTrainingResult = {
    val rep = trainingData.head
    val repLabels = rep.label
    val numLabels = repLabels.length
    assert(
      repLabels.forall(_.isInstanceOf[Double]),
      "Multi-task linear regression can only be performed for real-valued labels."
    )

    // Train each task independently so that label sparsity does not diminish the training corpus
    val singleTaskLearner = LinearRegressionLearner(regParam, fitIntercept)
    val singleTaskResults = Vector.tabulate(numLabels) { idx =>
      val reducedData = trainingData
        .map(row => row.mapLabel(labelVec => labelVec(idx).asInstanceOf[Double]))
        .filterNot(_.label.isNaN)
      singleTaskLearner.train(reducedData, rng)
    }
    MultiTaskLinearRegressionTrainingResult(singleTaskResults)
  }
}

case class MultiTaskLinearRegressionTrainingResult(linearResults: Vector[LinearRegressionTrainingResult])
    extends MultiTaskTrainingResult {

  override val models: Vector[LinearRegressionModel] = linearResults.map(_.model)

  override val model: MultiTaskModel = ParallelModels(models, Vector.fill(models.length)(true))

  override lazy val featureImportance: Option[Vector[Double]] = {
    // Average the importance across labels
    val labelImportance = linearResults.flatMap(_.featureImportance) // (# labels) x (# features) array
    Option.when(labelImportance.nonEmpty) {
      val importanceByFeature = labelImportance.transpose.map(_.sum)
      val totalImportance = importanceByFeature.sum
      importanceByFeature.map(_ / totalImportance)
    }
  }
}
