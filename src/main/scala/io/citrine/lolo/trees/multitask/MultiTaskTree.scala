package io.citrine.lolo.trees.multitask

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.trees.ModelNode
import io.citrine.lolo.trees.classification.ClassificationTree
import io.citrine.lolo.trees.regression.RegressionTree
import io.citrine.lolo.{Model, MultiTaskLearner, MultiTaskTrainingResult, ParallelModels, PredictionResult}

import scala.util.Random

/** A trait to hold logic common to all tree learners that operate on multiple labels. */
case class MultiTaskTreeLearner(
                                 randomizePivotLocation: Boolean = false,
                                 rng: Random = Random
                               ) extends MultiTaskLearner {

  /**
    * Construct one regression or classification tree for each label.
    *
    * @param trainingData   to train on
    * @param weights  for the training rows, if applicable
    * @return         sequence of models, one for each label
    */
  override def train(trainingData: Seq[(Vector[Any], Vector[Any])], weights: Option[Seq[Double]]): MultiTaskTreeTrainingResult = {
    val inputs = trainingData.map(_._1)
    val repInput = inputs.head
    val labels = trainingData.map(_._2)
    val repOutput = labels.head
    val labelIndices = repOutput.indices

    /* Create encoders for any categorical features */
    val inputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(inputs.map(_ (i))))
      }
    }
    val encodedInputs = inputs.map(r => CategoricalEncoder.encodeInput(r, inputEncoders))

    /* Create encoders for any categorical labels */
    val outputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repOutput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(labels(i).filterNot(_ == null)))
      }
    }
    val encodedLabels = labels.map(CategoricalEncoder.encodeInput(_, outputEncoders))

    // Encode the inputs, outputs, and filter out zero weight rows
    val collectedData = inputs.indices.map { i =>
      (encodedInputs(i), encodedLabels(i).toArray, weights.map(_ (i)).getOrElse(1.0))
    }.filter(_._3 > 0.0)

    // Construct the training tree
    val root = new MultiTaskTrainingNode(collectedData, randomizePivotLocation, rng)

    // Construct the model trees
    val nodes = labelIndices.map(root.getNode)

    // Stick the model trees into RegressionTree and ClassificationTree objects
    val models = labelIndices.map { i =>
      if (repOutput(i).isInstanceOf[Double]) {
        new RegressionTree(
          nodes(i).asInstanceOf[ModelNode[PredictionResult[Double]]],
          inputEncoders
        )
      } else {
        new ClassificationTree(
          nodes(i).asInstanceOf[ModelNode[PredictionResult[Char]]],
          inputEncoders,
          outputEncoders(i).get
        )
      }
    }

    val sumFeatureImportance: Vector[Double] = {
      val startingImportances = Vector.fill(repInput.length)(0.0)
      labelIndices.foldLeft(startingImportances) { (importance, i) =>
        root.getFeatureImportance(i).toVector.zip(importance).map(p => p._1 + p._2)
      }
    }

    new MultiTaskTreeTrainingResult(models, sumFeatureImportance)
  }
}

class MultiTaskTreeTrainingResult(
                                   models: Seq[Model[PredictionResult[Any]]],
                                   featureImportance: Vector[Double]
                                 ) extends MultiTaskTrainingResult {
  val model = new ParallelModels(models, models.map(_.isInstanceOf[RegressionTree]))
  private lazy val importanceNormalized = {
    if (Math.abs(featureImportance.sum) > 0) {
      featureImportance.map(_ / featureImportance.sum)
    } else {
      featureImportance.map(_ => 1.0 / featureImportance.size)
    }
  }

  override def getModel(): ParallelModels = model

  override def getModels(): Seq[Model[PredictionResult[Any]]] = models

  override def getFeatureImportance(): Option[Vector[Double]] = Some(importanceNormalized)

}
