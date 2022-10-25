package io.citrine.lolo.trees.multitask

import io.citrine.lolo.{PredictionResult, TrainingRow}
import io.citrine.lolo.linear.{GuessTheMeanLearner, GuessTheModeLearner}
import io.citrine.lolo.trees.classification.ClassificationTrainingLeaf
import io.citrine.lolo.trees.regression.RegressionTrainingLeaf
import io.citrine.lolo.trees.splits.{MultiTaskSplitter, NoSplit, Split}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.random.Random

import scala.collection.mutable

case class MultiTaskTrainingNode(
    trainingData: Seq[TrainingRow[Vector[AnyVal]]],
    labelWiseInstructions: Seq[MultiTaskLabelInstruction],
    deltaImpurity: Double
) {

  // get feature importance for the `index`th label
  def featureImportanceByLabelIndex(index: Int): mutable.ArraySeq[Double] = {
    labelWiseInstructions(index) match {
      case Stop(leaf)             => leaf.featureImportance
      case FollowChild(childNode) => childNode.featureImportanceByLabelIndex(index)
      case DoSplit(split, leftNode, rightNode) =>
        val ans = leftNode
          .featureImportanceByLabelIndex(index)
          .zip(rightNode.featureImportanceByLabelIndex(index))
          .map(p => p._1 + p._2)
        ans(split.index) = ans(split.index) + deltaImpurity
        ans
      case Inaccessible() => throw new RuntimeException(s"No valid training data present for label $index")
    }
  }

  // Construct the model node for the `index`th label
  def modelNodeByLabelIndex(index: Int): ModelNode[Any] = {
    labelWiseInstructions(index) match {
      case Stop(leaf)             => leaf.modelNode
      case FollowChild(childNode) => childNode.modelNodeByLabelIndex(index)
      case DoSplit(split, leftNode, rightNode) =>
        if (trainingData.head.label(index).isInstanceOf[Double]) {
          new InternalModelNode[PredictionResult[Double]](
            split = split,
            left = leftNode.modelNodeByLabelIndex(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
            right = rightNode.modelNodeByLabelIndex(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
            outputDimension = 0, // Shapley is not implemented for multi-task nodes
            trainingWeight = trainingData.map(_.weight).sum
          )
        } else {
          new InternalModelNode[PredictionResult[Char]](
            split = split,
            left = leftNode.modelNodeByLabelIndex(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
            right = rightNode.modelNodeByLabelIndex(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
            outputDimension = 0,
            trainingWeight = trainingData.map(_.weight).sum
          )
        }
      case Inaccessible() => throw new RuntimeException(s"No valid training data present for label $index")
    }
  }
}

object MultiTaskTrainingNode {

  /**
    * Build a node by computing splits and recursively building the child nodes.
    *
    * @param trainingData on which to build the split
    * @param numFeatures to considering when determining the split
    * @param remainingDepth the maximum number of splits left
    * @param maxDepth maximum number of splits (used to compute depth of this node)
    * @param minInstances minimum training instances per node
    * @param splitter determines the best split
    * @param rng random number generator, for reproducibility
    * @return
    */
  def build(
      trainingData: Seq[TrainingRow[Vector[AnyVal]]],
      numFeatures: Int,
      remainingDepth: Int,
      maxDepth: Int,
      minInstances: Int,
      splitter: MultiTaskSplitter,
      rng: Random = Random()
  ): MultiTaskTrainingNode = {
    // Determine the overall split.
    val sufficientData = trainingData.size >= 2 * minInstances &&
      remainingDepth > 0 &&
      trainingData.exists(row => row.label != trainingData.head.label)
    val (split: Split, deltaImpurity: Double) = if (sufficientData) {
      splitter.getBestSplit(
        trainingData,
        numFeatures,
        minInstances,
        rng
      )
    } else {
      (NoSplit(), 0.0)
    }
    // Build the left and right nodes (assuming the split exists).
    val nodesOpt: Option[(MultiTaskTrainingNode, MultiTaskTrainingNode)] = split match {
      case _: NoSplit => None
      case split: Split =>
        val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r.inputs))
        val leftNode = MultiTaskTrainingNode.build(
          trainingData = leftTrain,
          numFeatures = numFeatures,
          remainingDepth = remainingDepth - 1,
          maxDepth = maxDepth,
          minInstances = minInstances,
          splitter = splitter,
          rng = rng
        )
        val rightNode = MultiTaskTrainingNode.build(
          trainingData = rightTrain,
          numFeatures = numFeatures,
          remainingDepth = remainingDepth - 1,
          maxDepth = maxDepth,
          minInstances = minInstances,
          splitter = splitter,
          rng = rng
        )
        Some((leftNode, rightNode))
    }
    // Determine what to do for each label (evaluate a model, evaluate the split, go left, or go right).
    val exampleRow = trainingData.head
    val labelWiseInstructions = exampleRow.label.indices.map { index =>
      // Determine how much data *with this label* goes down each branch
      val exampleLabel = exampleRow.label(index)
      val reducedData: Seq[TrainingRow[Any]] = if (exampleLabel.isInstanceOf[Double]) {
        trainingData.map(_.mapLabel(labelVec => labelVec(index).asInstanceOf[Double])).filterNot(_.label.isNaN)
      } else {
        trainingData.map(_.mapLabel(labelVec => labelVec(index).asInstanceOf[Char])).filter(_.label > 0)
      }
      val (left, right) = reducedData.partition(r => split.turnLeft(r.inputs))

      if (reducedData.isEmpty) {
        Inaccessible()
      } else if (nodesOpt.isEmpty || reducedData.length <= minInstances) {
        // Either there's no split or there's not enough data remaining with this label. Build a leaf node.
        val trainingLeaf: TrainingNode[AnyVal] = if (exampleLabel.isInstanceOf[Double]) {
          RegressionTrainingLeaf
            .build(
              reducedData.asInstanceOf[Seq[TrainingRow[Double]]],
              GuessTheMeanLearner(),
              maxDepth - remainingDepth,
              rng
            )
        } else {
          ClassificationTrainingLeaf
            .build(
              reducedData.asInstanceOf[Seq[TrainingRow[Char]]],
              GuessTheModeLearner(),
              maxDepth - remainingDepth,
              rng
            )
        }
        Stop(trainingLeaf)
      } else if (left.nonEmpty && right.nonEmpty) {
        // From this point onwards we know that nodesOpt is defined. Follow the split.
        DoSplit(split, nodesOpt.get._1, nodesOpt.get._2)
      } else if (left.nonEmpty) {
        // This label only has data on the left-hand side, so go left.
        FollowChild(nodesOpt.get._1)
      } else {
        // This label only has data on the right-hand side, so go right.
        FollowChild(nodesOpt.get._2)
      }
    }

    MultiTaskTrainingNode(
      trainingData = trainingData,
      labelWiseInstructions = labelWiseInstructions,
      deltaImpurity = deltaImpurity
    )
  }
}

/** An enumeration specifying what behavior to perform for a specific label. */
sealed trait MultiTaskLabelInstruction

/** Follow the proscribed split. */
case class DoSplit(
    split: Split,
    leftNode: MultiTaskTrainingNode,
    rightNode: MultiTaskTrainingNode
) extends MultiTaskLabelInstruction

/** Always go to the specified child node. */
case class FollowChild(childNode: MultiTaskTrainingNode) extends MultiTaskLabelInstruction

/** Stop and evaluate the provided leaf node. */
case class Stop(leaf: TrainingNode[AnyVal]) extends MultiTaskLabelInstruction

/** A placeholder for inaccessible sections of the tree, such as underneath a Stop() instruction. */
case class Inaccessible() extends MultiTaskLabelInstruction
