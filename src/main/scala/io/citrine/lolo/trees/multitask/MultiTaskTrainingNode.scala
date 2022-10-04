package io.citrine.lolo.trees.multitask

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.classification.ClassificationTrainingLeaf
import io.citrine.lolo.trees.regression.RegressionTrainingLeaf
import io.citrine.lolo.trees.splits.{MultiTaskSplitter, NoSplit, Split}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.random.Random

import scala.collection.mutable

case class MultiTaskTrainingNode(
    trainingData: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
    labelWiseInstructions: Seq[MultiTaskLabelInstruction],
    split: Split,
    leftNodeOption: Option[MultiTaskTrainingNode],
    rightNodeOption: Option[MultiTaskTrainingNode],
    deltaImpurity: Double
) {

  // get feature importance for the i'th label
  def getFeatureImportance(index: Int): mutable.ArraySeq[Double] = {
    labelWiseInstructions(index) match {
      case Stop(leaf) => leaf.getFeatureImportance()
      case GoLeft()   => leftNodeOption.get.getFeatureImportance(index)
      case GoRight()  => rightNodeOption.get.getFeatureImportance(index)
      case DoSplit() =>
        val ans = leftNodeOption.get
          .getFeatureImportance(index)
          .zip(rightNodeOption.get.getFeatureImportance(index))
          .map(p => p._1 + p._2)
        ans(split.index) = ans(split.index) + deltaImpurity
        ans
      case Inaccessible() => throw new RuntimeException(s"No valid training data present for label $index")
    }
  }

  // Construct the model node for the `index`th label
  def getNode(index: Int): ModelNode[PredictionResult[Any]] = {
    labelWiseInstructions(index) match {
      case Stop(leaf) => leaf.getModelNode()
      case GoLeft()   => leftNodeOption.get.getNode(index)
      case GoRight()  => rightNodeOption.get.getNode(index)
      case DoSplit() =>
        if (trainingData.head._2(index).isInstanceOf[Double]) {
          new InternalModelNode[PredictionResult[Double]](
            split = split,
            left = leftNodeOption.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
            right = rightNodeOption.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
            outputDimension = 0,
            trainingWeight = 0.0
          )
        } else {
          new InternalModelNode[PredictionResult[Char]](
            split = split,
            left = leftNodeOption.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
            right = rightNodeOption.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
            outputDimension = 0,
            trainingWeight = 0.0
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
      trainingData: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
      numFeatures: Int,
      remainingDepth: Int,
      maxDepth: Int,
      minInstances: Int,
      splitter: MultiTaskSplitter,
      rng: Random = Random()
  ): MultiTaskTrainingNode = {
    val sufficientData = trainingData.size >= 2 * minInstances &&
      remainingDepth > 0 &&
      trainingData.exists(row => !row._2.sameElements(trainingData.head._2))
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

    val exampleRow = trainingData.head
    val labelWiseInstructions = exampleRow._2.indices.map { index =>
      val exampleLabel = exampleRow._2(index)
      val reducedData = if (exampleLabel.isInstanceOf[Double]) {
        trainingData.map(x => (x._1, x._2(index).asInstanceOf[Double], x._3)).filterNot(_._2.isNaN)
      } else {
        trainingData.map(x => (x._1, x._2(index).asInstanceOf[Char], x._3)).filter(_._2 > 0)
      }
      val (left, right) = reducedData.partition(r => split.turnLeft(r._1))

      if (reducedData.isEmpty) {
        Inaccessible()
      } else if (split.isInstanceOf[NoSplit] || reducedData.length <= minInstances) {
        val trainingLeaf: TrainingNode[AnyVal] = if (exampleLabel.isInstanceOf[Double]) {
          RegressionTrainingLeaf
            .build(
              reducedData.asInstanceOf[Seq[(Vector[AnyVal], Double, Double)]],
              GuessTheMeanLearner(),
              maxDepth - remainingDepth,
              rng
            )
        } else {
          ClassificationTrainingLeaf
            .build(
              reducedData.asInstanceOf[Seq[(Vector[AnyVal], Char, Double)]],
              GuessTheMeanLearner(),
              maxDepth - remainingDepth,
              rng
            )
        }
        Stop(trainingLeaf)
      } else if (left.nonEmpty && right.nonEmpty) {
        DoSplit()
      } else if (left.nonEmpty) {
        GoLeft()
      } else {
        GoRight()
      }
    }

    split match {
      case split: NoSplit =>
        MultiTaskTrainingNode(
          trainingData = trainingData,
          labelWiseInstructions = labelWiseInstructions,
          split = split,
          leftNodeOption = None,
          rightNodeOption = None,
          deltaImpurity = deltaImpurity
        )
      case split: Split =>
        val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
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
        MultiTaskTrainingNode(
          trainingData = trainingData,
          labelWiseInstructions = labelWiseInstructions,
          split = split,
          leftNodeOption = Some(leftNode),
          rightNodeOption = Some(rightNode),
          deltaImpurity = deltaImpurity
        )
    }
  }
}

/**
  * An enumeration specifying what behavior to perform for a specific label.
  * The logic of MultiTaskTrainingNode.build() enforces that instructions will only be given in valid situations.
  * For example, DoSplit() will only occur if there is a valid split and this label has training data that
  * go down both paths.
  */
sealed trait MultiTaskLabelInstruction

/** Follow the proscribed split. */
case class DoSplit() extends MultiTaskLabelInstruction

/** Always go to the left child. */
case class GoLeft() extends MultiTaskLabelInstruction

/** Always go to the riht child. */
case class GoRight() extends MultiTaskLabelInstruction

/** Stop and evaluate the provided leaf node. */
case class Stop(leaf: TrainingNode[AnyVal]) extends MultiTaskLabelInstruction

/** A placeholder for inaccessible sections of the tree, such as underneath a Stop() instruction. */
case class Inaccessible() extends MultiTaskLabelInstruction
