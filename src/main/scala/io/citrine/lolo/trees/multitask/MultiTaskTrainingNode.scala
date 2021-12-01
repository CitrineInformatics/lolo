package io.citrine.lolo.trees.multitask

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.splits.{MultiTaskSplitter, NoSplit}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingLeaf}

import scala.collection.mutable
import scala.util.Random

/**
  * Node in a multi-task training tree, which can produce nodes for its model trees
  *
  * @param inputs data on which to select splits and form models
  */
class MultiTaskTrainingNode(
                             inputs: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
                             randomizePivotLocation: Boolean = false,
                             rng: Random = Random
                           ) {

  // Compute a split
  val (split, deltaImpurity) = MultiTaskSplitter(rng = rng, randomizePivotLocation = randomizePivotLocation)
    .getBestSplit(inputs, numFeatures = inputs.head._1.size, minInstances = 1)

  // Try to construct left and right children
  val (leftChild: Option[MultiTaskTrainingNode], rightChild: Option[MultiTaskTrainingNode]) = split match {
    case _: NoSplit => (None, None)
    case _: Any =>
      val (leftData, rightData) = inputs.partition(row => split.turnLeft(row._1))
      (Some(new MultiTaskTrainingNode(leftData, randomizePivotLocation, rng = rng)), Some(new MultiTaskTrainingNode(rightData, randomizePivotLocation, rng = rng)))
  }

  // TODO: getFeatureImportance and getNode have lots of overlapping code. Combine them
  def getFeatureImportance(index: Int): mutable.ArraySeq[Double] = {
    // Filter out "missing" values, which are NaN for regression and 0 for encoded categoricals
    val label = inputs.head._2(index)
    val reducedData = if (label.isInstanceOf[Double]) {
      inputs.map(x => (x._1, x._2(index).asInstanceOf[Double], x._3)).filterNot(_._2.isNaN)
    } else {
      inputs.map(x => (x._1, x._2(index).asInstanceOf[Char], x._3)).filter(_._2 > 0)
    }
    // Compute the valid data for each child
    val (left, right) = reducedData.partition(r => split.turnLeft(r._1))

    (leftChild, rightChild) match {
      case (Some(theLeftChild), Some(theRightChild)) if left.nonEmpty && right.nonEmpty =>
        val ans = theLeftChild.getFeatureImportance(index).zip(theRightChild.getFeatureImportance(index)).map(p => p._1 + p._2)
        ans(split.getIndex()) = ans(split.getIndex()) + deltaImpurity
        ans
      case (Some(theLeftChild), _) if left.nonEmpty =>
        theLeftChild.getFeatureImportance(index)
      case (_, Some(theRightChild)) if right.nonEmpty =>
        theRightChild.getFeatureImportance(index)
      case (_, _) =>
        if (label.isInstanceOf[Double]) {
          new TrainingLeaf[Double](reducedData.asInstanceOf[Seq[(Vector[AnyVal], Double, Double)]], new GuessTheMeanLearner(), 1).getFeatureImportance()
        } else {
          new TrainingLeaf[Char](reducedData.asInstanceOf[Seq[(Vector[AnyVal], Char, Double)]], new GuessTheMeanLearner(), 1).getFeatureImportance()
        }
    }
  }

  // Construct the model node for the `index`th label
  def getNode(index: Int): ModelNode[PredictionResult[Any]] = {

    // Filter out "missing" values, which are NaN for regression and 0 for encoded categoricals
    val label = inputs.head._2(index)
    val reducedData = if (label.isInstanceOf[Double]) {
      inputs.map(x => (x._1, x._2(index).asInstanceOf[Double], x._3)).filterNot(_._2.isNaN)
    } else {
      inputs.map(x => (x._1, x._2(index).asInstanceOf[Char], x._3)).filter(_._2 > 0)
    }
    // Compute the valid data for each child
    val (left, right) = reducedData.partition(r => split.turnLeft(r._1))

    // Construct an internal node if the children are defined and actually have valid data
    val node = if (leftChild.isDefined && rightChild.isDefined && left.nonEmpty && right.nonEmpty) {
      // Not a case because of erasure :-(
      if (label.isInstanceOf[Double]) {
        new InternalModelNode[PredictionResult[Double]](
          split,
          leftChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
          rightChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
          outputDimension = 0,  // Don't support multitask SHAP at this time.
          trainingWeight = reducedData.length.toDouble
        )
      } else {
        if (!label.isInstanceOf[Char]) throw new IllegalArgumentException("Training data wasn't double or char")
        new InternalModelNode[PredictionResult[Char]](
          split,
          leftChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
          rightChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
          outputDimension = 0,  // Don't support multitask SHAP at this time.
          trainingWeight = reducedData.length.toDouble
        )
      }
    } else if (leftChild.isDefined && left.nonEmpty) {
      leftChild.get.getNode(index)
    } else if (rightChild.isDefined && right.nonEmpty) {
      rightChild.get.getNode(index)
    } else {
      // If there are no children or the children don't have valid data for this label, emit a leaf (with GTM)
      if (label.isInstanceOf[Double]) {
        // TODO: this can be a RegressionTrainingLeaf, and the one below can maybe be a ClassificaitonTrainingLeaf
        new TrainingLeaf[Double](reducedData.asInstanceOf[Seq[(Vector[AnyVal], Double, Double)]], GuessTheMeanLearner(), 1).getNode()
      } else {
        new TrainingLeaf[Char](reducedData.asInstanceOf[Seq[(Vector[AnyVal], Char, Double)]], GuessTheMeanLearner(), 1).getNode()
      }
    }
    node.asInstanceOf[ModelNode[PredictionResult[Any]]]
  }
}
