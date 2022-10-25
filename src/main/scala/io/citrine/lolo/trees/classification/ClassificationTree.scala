package io.citrine.lolo.trees.classification

import io.citrine.random.Random
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.linear.GuessTheModeLearner
import io.citrine.lolo.trees.splits.{ClassificationSplitter, Splitter}
import io.citrine.lolo.trees.{ModelNode, TrainingNode, TreeMeta}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult, TrainingRow}

import scala.collection.mutable

/**
  * @param numFeatures subset of features to select splits from
  * @param maxDepth maximum depth of tree
  * @param minLeafInstances minimum training instances per node
  * @param leafLearner to train on leaves
  * @param splitter used to select splits
  */
case class ClassificationTreeLearner(
    numFeatures: Int = -1,
    maxDepth: Int = 30,
    minLeafInstances: Int = 1,
    leafLearner: Option[Learner[Char]] = None,
    splitter: Splitter[Char] = ClassificationSplitter()
) extends Learner[Any] {

  @transient private lazy val myLeafLearner: Learner[Char] = leafLearner.getOrElse(GuessTheModeLearner())

  /**
    * Train a classification tree.
    *
    * @param trainingData to train on
    * @param rng          random number generator for reproducibility
    * @return a classification tree
    */
  override def train(trainingData: Seq[TrainingRow[Any]], rng: Random): ClassificationTreeTrainingResult = {
    assert(trainingData.size > 4, s"We need to have at least 4 rows, only ${trainingData.size} given")
    val repInput = trainingData.head.inputs

    // Create encoders for any categorical features
    val inputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map {
      case (v, i) =>
        if (v.isInstanceOf[Double]) {
          None
        } else {
          Some(CategoricalEncoder.buildEncoder(trainingData.map(_.inputs(i))))
        }
    }

    val outputEncoder = CategoricalEncoder.buildEncoder(trainingData.map(_.label))

    // Encode the training data
    val encodedTraining = trainingData.map { row =>
      val encodedInputs = CategoricalEncoder.encodeInput(row.inputs, inputEncoders)
      val encodedLabels = outputEncoder.encode(row.label)
      TrainingRow(encodedInputs, encodedLabels, row.weight)
    }

    /* Add the weights to the (features, label) tuples and remove any with zero weight */
    val finalTraining = encodedTraining.filter(_.weight > 0.0)

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head.inputs.size
    }

    // Recursively build the tree via its nodes and wrap the top node in a ClassificationTreeTrainingResult
    val rootTrainingNode = ClassificationTrainingNode.build(
      trainingData = finalTraining,
      leafLearner = myLeafLearner,
      splitter = splitter,
      numFeatures = numFeaturesActual,
      minLeafInstances = minLeafInstances,
      remainingDepth = maxDepth,
      maxDepth = maxDepth,
      numClasses = trainingData.map(_.label).distinct.length,
      rng = rng
    )
    ClassificationTreeTrainingResult(rootTrainingNode, inputEncoders, outputEncoder)
  }
}

case class ClassificationTreeTrainingResult(
    rootTrainingNode: TrainingNode[Char],
    inputEncoders: Seq[Option[CategoricalEncoder[Any]]],
    outputEncoder: CategoricalEncoder[Any]
) extends TrainingResult[Any] {

  // Grab a prediction node. The partitioning happens here
  override lazy val model: ClassificationTree =
    ClassificationTree(rootTrainingNode.modelNode, inputEncoders, outputEncoder)

  // Grab the feature influences
  lazy val nodeImportance: mutable.ArraySeq[Double] = rootTrainingNode.featureImportance

  override lazy val featureImportance: Option[Vector[Double]] = Some(
    if (Math.abs(nodeImportance.sum) > 0) {
      nodeImportance.map(_ / nodeImportance.sum).toVector
    } else {
      nodeImportance.map(_ => 1.0 / nodeImportance.size).toVector
    }
  )
}

/**
  * Classification tree
  */
case class ClassificationTree(
    rootModelNode: ModelNode[Char],
    inputEncoders: Seq[Option[CategoricalEncoder[Any]]],
    outputEncoder: CategoricalEncoder[Any]
) extends Model[Any] {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): ClassificationTreePrediction = {
    ClassificationTreePrediction(
      inputs.map(inp => rootModelNode.transform(CategoricalEncoder.encodeInput(inp, inputEncoders))),
      outputEncoder
    )
  }
}

/**
  * Classification result
  */
case class ClassificationTreePrediction(
    predictions: Seq[(PredictionResult[Char], TreeMeta)],
    outputEncoder: CategoricalEncoder[Any]
) extends PredictionResult[Any] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def expected: Seq[Any] = predictions.map(p => outputEncoder.decode(p._1.expected.head))

  def depth: Seq[Int] = predictions.map(_._2.depth)
}
