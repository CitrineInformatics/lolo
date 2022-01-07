package io.citrine.lolo.trees.multitask

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.regression.RegressionTrainingLeaf
import io.citrine.lolo.trees.splits.{MultiTaskSplitter, NoSplit}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingLeaf}

import scala.collection.mutable
import scala.util.Random

/** Node in a multi-task training tree, which can produce nodes for its model trees. Splits are chosen using a
  * MultiTaskSplitter, which considers the sum impurity decrease across all tasks.
  *
  * @param inputs
  *   data on which to select splits and form models
  * @param numFeatures
  *   to randomly select from at each split (negative values indicate that all features should be considered)
  * @param maxDepth
  *   to grow the tree to
  * @param minInstances
  *   minimum number of training instances per leaf
  * @param randomizePivotLocation
  *   whether to generate splits randomly between the data points
  * @param rng
  *   random number generator, for reproducibility
  */
class MultiTaskTrainingNode(
    inputs: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
    numFeatures: Int,
    maxDepth: Int,
    minInstances: Int,
    randomizePivotLocation: Boolean = false,
    rng: Random = Random
) {

  // Compute a split
  val (split, deltaImpurity) = if (maxDepth <= 0) {
    (new NoSplit, 0.0)
  } else {
    MultiTaskSplitter(rng = rng, randomizePivotLocation = randomizePivotLocation).getBestSplit(
      inputs,
      numFeatures,
      minInstances
    )
  }

  // Try to construct left and right children
  val (leftChild: Option[MultiTaskTrainingNode], rightChild: Option[MultiTaskTrainingNode]) = split match {
    case _: NoSplit => (None, None)
    case _: Any =>
      val (leftData, rightData) = inputs.partition(row => split.turnLeft(row._1))
      (
        Some(
          new MultiTaskTrainingNode(
            leftData,
            numFeatures,
            maxDepth - 1,
            minInstances,
            randomizePivotLocation,
            rng = rng
          )
        ),
        Some(
          new MultiTaskTrainingNode(
            rightData,
            numFeatures,
            maxDepth - 1,
            minInstances,
            randomizePivotLocation,
            rng = rng
          )
        )
      )
  }

  // get feature importance for the i'th label
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

    // get feature importance from the children if they exist, or from this node if it is a leaf
    (leftChild, rightChild) match {
      case (Some(theLeftChild), Some(theRightChild)) if left.nonEmpty && right.nonEmpty =>
        val ans =
          theLeftChild.getFeatureImportance(index).zip(theRightChild.getFeatureImportance(index)).map(p => p._1 + p._2)
        ans(split.getIndex()) = ans(split.getIndex()) + deltaImpurity
        ans
      case (Some(theLeftChild), _) if left.nonEmpty =>
        theLeftChild.getFeatureImportance(index)
      case (_, Some(theRightChild)) if right.nonEmpty =>
        theRightChild.getFeatureImportance(index)
      case (_, _) =>
        if (label.isInstanceOf[Double]) {
          new RegressionTrainingLeaf(
            reducedData.asInstanceOf[Seq[(Vector[AnyVal], Double, Double)]],
            GuessTheMeanLearner(),
            1
          ).getFeatureImportance()
        } else {
          new TrainingLeaf[Char](
            reducedData.asInstanceOf[Seq[(Vector[AnyVal], Char, Double)]],
            GuessTheMeanLearner(),
            1
          ).getFeatureImportance()
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
    (leftChild, rightChild) match {
      case (Some(theLeftChild), Some(theRightChild)) if left.nonEmpty && right.nonEmpty =>
        if (label.isInstanceOf[Double]) {
          new InternalModelNode[PredictionResult[Double]](
            split,
            theLeftChild.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
            theRightChild.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
            outputDimension = 0, // Don't support multitask SHAP at this time.
            trainingWeight = reducedData.length.toDouble
          )
        } else {
          if (!label.isInstanceOf[Char]) throw new IllegalArgumentException("Training data wasn't double or char")
          new InternalModelNode[PredictionResult[Char]](
            split,
            theLeftChild.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
            theRightChild.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
            outputDimension = 0, // Don't support multitask SHAP at this time.
            trainingWeight = reducedData.length.toDouble
          )
        }
      case (Some(theLeftChild), _) if left.nonEmpty =>
        theLeftChild.getNode(index)
      case (_, Some(theRightChild)) if right.nonEmpty =>
        theRightChild.getNode(index)
      case (_, _) =>
        if (label.isInstanceOf[Double]) {
          new RegressionTrainingLeaf(
            reducedData.asInstanceOf[Seq[(Vector[AnyVal], Double, Double)]],
            GuessTheMeanLearner(),
            1
          ).getNode()
        } else {
          new TrainingLeaf[Char](
            reducedData.asInstanceOf[Seq[(Vector[AnyVal], Char, Double)]],
            GuessTheMeanLearner(),
            1
          ).getNode()
        }
    }
  }
}
