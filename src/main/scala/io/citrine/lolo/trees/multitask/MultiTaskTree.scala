package io.citrine.lolo.trees.multitask

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.trees.ModelNode
import io.citrine.lolo.trees.classification.ClassificationTree
import io.citrine.lolo.trees.regression.RegressionTree
import io.citrine.lolo.{Model, MultiTaskLearner, PredictionResult, TrainingResult}

class MultiTaskTreeLearner extends MultiTaskLearner {

  /**
    * Train a model
    *
    * @param inputs  to train on
    * @param labels  sequence of sequences of labels
    * @param weights for the training rows, if applicable
    * @return training result containing a model
    */
  override def train(inputs: Seq[Vector[Any]], labels: Seq[Seq[Any]], weights: Option[Seq[Double]]): Seq[TrainingResult] = {
    val labelsTransposed = labels.toVector.transpose

    /* Create encoders for any categorical features */
    val repInput = inputs.head
    val inputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(inputs.map(_(i))))
      }
    }
    val encodedInputs = inputs.map(r => CategoricalEncoder.encodeInput(r, inputEncoders))

    val repOutput = labels.map(_.head)
    val outputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repOutput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(labels(i)))
      }
    }
    val encodedLabels = labelsTransposed.map(CategoricalEncoder.encodeInput(_, outputEncoders))

    val collectedData = inputs.indices.map{i =>
      (encodedInputs(i), encodedLabels(i).toArray, weights.map(_(i)).getOrElse(1.0))
    }
    val root = new MultiTaskTrainingNode(collectedData)
    val nodes = root.getNodes()

    val models = labels.indices.map{i =>
      if (labels(i).head.isInstanceOf[Double]) {
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

    models.map(new MultiTaskTreeTrainingResult(_, hypers))
  }
}

class MultiTaskTreeTrainingResult(model: Model[PredictionResult[Any]], hypers: Map[String, Any]) extends TrainingResult {
  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[PredictionResult[Any]] = model

  /**
    * Get the hyperparameters used to train this model
    *
    * @return hypers set for model
    */
  override def getHypers(): Map[String, Any] = hypers
}
