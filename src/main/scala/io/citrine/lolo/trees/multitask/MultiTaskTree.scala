package io.citrine.lolo.trees.multitask

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.trees.ModelNode
import io.citrine.lolo.trees.classification.ClassificationTree
import io.citrine.lolo.trees.regression.RegressionTree
import io.citrine.lolo.{Model, MultiTaskLearner, PredictionResult, TrainingResult}

import scala.util.Random

/**
  * Multi-task tree learner, which produces multiple decision trees with the same split structure
  *
  * @param randomizePivotLocation whether to generate splits randomly between the data points
  * @param rng                    random number generator to use
  * @param combinedModel          whether to create a single model that predicts all labels or a sequence of models,
  *                               one for each label. Creating one model allows for correlated uncertainty predictions.
  */
case class MultiTaskTreeLearner(
                                 randomizePivotLocation: Boolean = false,
                                 rng: Random = Random,
                                 combinedModel: Boolean = false
                               ) extends MultiTaskLearner {

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
        Some(CategoricalEncoder.buildEncoder(inputs.map(_ (i))))
      }
    }
    val encodedInputs = inputs.map(r => CategoricalEncoder.encodeInput(r, inputEncoders))

    /* Create encoders for any categorical labels */
    val repOutput = labels.map(_.head)
    val outputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repOutput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(labels(i).filterNot(_ == null)))
      }
    }
    val encodedLabels = labelsTransposed.map(CategoricalEncoder.encodeInput(_, outputEncoders))

    // Encode the inputs, outputs, and filter out zero weight rows
    val collectedData = inputs.indices.map { i =>
      (encodedInputs(i), encodedLabels(i).toArray, weights.map(_ (i)).getOrElse(1.0))
    }.filter(_._3 > 0.0)

    // Construct the training tree
    val root = new MultiTaskTrainingNode(collectedData, randomizePivotLocation, rng)

    // Construct the model trees
    val nodes = labels.indices.map(root.getNode)

    // Stick the model trees into RegressionTree and ClassificationTree objects
    val models = labels.indices.map { i =>
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

    // Wrap the models in dead-simple training results and return
    if (combinedModel) {
      Seq(new MultiTaskTreeParallelTrainingResult(models))
    } else {
      models.map(new MultiTaskTreeTrainingResult(_))
    }
  }
}

class MultiTaskTreeParallelTrainingResult(models: Seq[Model[PredictionResult[Any]]]) extends TrainingResult {
  val model = new ParallelModels(models, models.map(_.isInstanceOf[RegressionTree]))

  override def getModel(): ParallelModels = model

  // TODO: combine feature importances of individual models (which are not currently available)
}

class MultiTaskTreeTrainingResult(model: Model[PredictionResult[Any]]) extends TrainingResult {
  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  override def getModel(): Model[PredictionResult[Any]] = model
}
