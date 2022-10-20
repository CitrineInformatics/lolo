package io.citrine.lolo.trees.multitask

import io.citrine.random.Random
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.trees.ModelNode
import io.citrine.lolo.trees.classification.ClassificationTree
import io.citrine.lolo.trees.regression.RegressionTree
import io.citrine.lolo.trees.splits.MultiTaskSplitter
import io.citrine.lolo.{Model, MultiTaskLearner, MultiTaskTrainingResult, ParallelModels, TrainingRow}

/**
  * A tree learner that operates on multiple labels.
  *
  * @param numFeatures to random select from at each split (numbers less than 0 indicate that all features are used)
  * @param maxDepth to grow the tree to
  * @param minLeafInstances minimum number of training instances per leaf
  * @param splitter to determine the best split given data
  */
case class MultiTaskTreeLearner(
    numFeatures: Int = -1,
    maxDepth: Int = 30,
    minLeafInstances: Int = 1,
    splitter: MultiTaskSplitter = MultiTaskSplitter(randomizePivotLocation = true)
) extends MultiTaskLearner {

  /**
    * Construct one regression or classification tree for each label.
    *
    * @param trainingData to train on
    * @param rng          random number generator for reproducibility
    * @return             sequence of models, one for each label
    */
  override def train(trainingData: Seq[TrainingRow[Vector[Any]]], rng: Random): MultiTaskTreeTrainingResult = {
    val (inputs, labels, weights) = trainingData.map(_.asTuple).unzip3
    val repInput = inputs.head
    val repOutput = labels.head
    val labelIndices = repOutput.indices

    /* Create encoders for any categorical features */
    val inputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map {
      case (v, i) =>
        if (v.isInstanceOf[Double]) {
          None
        } else {
          Some(CategoricalEncoder.buildEncoder(inputs.map(_(i))))
        }
    }
    val encodedInputs = inputs.map(r => CategoricalEncoder.encodeInput(r, inputEncoders))

    /* Create encoders for any categorical labels */
    val outputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repOutput.zipWithIndex.map {
      case (v, i) =>
        if (v.isInstanceOf[Double]) {
          None
        } else {
          Some(CategoricalEncoder.buildEncoder(labels.map(_(i)).filterNot(_ == null)))
        }
    }
    val encodedLabels = labels.map(CategoricalEncoder.encodeInput(_, outputEncoders))

    // Encode the inputs, outputs, and filter out zero weight rows
    val collectedData = inputs.indices
      .map { i => TrainingRow(encodedInputs(i), encodedLabels(i), weights(i)) }
      .filter(_.weight > 0.0)

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      collectedData.head.inputs.size
    }

    // Construct the training tree
    val root = MultiTaskTrainingNode.build(
      trainingData = collectedData,
      numFeatures = numFeaturesActual,
      remainingDepth = maxDepth,
      maxDepth = maxDepth,
      minInstances = minLeafInstances,
      splitter = splitter,
      rng = rng
    )

    // Construct the model trees
    val nodes = labelIndices.map(root.modelNodeByLabelIndex)

    // Stick the model trees into RegressionTree and ClassificationTree objects
    val models = labelIndices.map { i =>
      if (repOutput(i).isInstanceOf[Double]) {
        new RegressionTree(
          nodes(i).asInstanceOf[ModelNode[Double]],
          inputEncoders
        )
      } else {
        new ClassificationTree(
          nodes(i).asInstanceOf[ModelNode[Char]],
          inputEncoders,
          outputEncoders(i).get
        )
      }
    }

    val sumFeatureImportance: Vector[Double] = {
      val startingImportances = Vector.fill(repInput.length)(0.0)
      labelIndices.foldLeft(startingImportances) { (importance, i) =>
        root.featureImportanceByLabelIndex(i).toVector.zip(importance).map(p => p._1 + p._2)
      }
    }

    new MultiTaskTreeTrainingResult(models, sumFeatureImportance)
  }
}

class MultiTaskTreeTrainingResult(
    override val models: Seq[Model[Any]],
    nodeImportance: Vector[Double]
) extends MultiTaskTrainingResult {

  override val model: ParallelModels = ParallelModels(models, models.map(_.isInstanceOf[RegressionTree]))

  override lazy val featureImportance: Option[Vector[Double]] = Some(
    if (Math.abs(nodeImportance.sum) > 0) {
      nodeImportance.map(_ / nodeImportance.sum)
    } else {
      nodeImportance.map(_ => 1.0 / nodeImportance.size)
    }
  )
}
