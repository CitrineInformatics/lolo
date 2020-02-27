package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.splits.{NoSplit, RegressionSplitter, Split, Splitter}
import io.citrine.lolo.trees.{FeaturePath, InternalModelNode, ModelNode, TrainingLeaf, TrainingNode}
import io.citrine.lolo.{Learner, PredictionResult}

/**
  * Created by maxhutch on 1/12/17.
  */
class RegressionTrainingNode(
                              trainingData: Seq[(Vector[AnyVal], Double, Double)],
                              leafLearner: Learner,
                              splitter: Splitter[Double],
                              split: Split,
                              deltaImpurity: Double,
                              numFeatures: Int,
                              minLeafInstances: Int,
                              remainingDepth: Int,
                              maxDepth: Int
                            )
  extends TrainingNode(
    trainingData = trainingData,
    remainingDepth = remainingDepth
  ) {

  // val (split: Split, deltaImpurity: Double) = RegressionSplitter.getBestSplit(trainingData, numFeatures)
  assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
  assert(!split.isInstanceOf[NoSplit], s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  assert(leftTrain.nonEmpty && rightTrain.nonEmpty, s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}")

  lazy val leftChild = RegressionTrainingNode.buildChild(leftTrain, leafLearner, splitter, minLeafInstances, remainingDepth, maxDepth, numFeatures)
  lazy val rightChild = RegressionTrainingNode.buildChild(rightTrain, leafLearner, splitter, minLeafInstances, remainingDepth, maxDepth, numFeatures)

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Double]] = {
    new InternalModelNode[PredictionResult[Double]](split, leftChild.getNode(), rightChild.getNode())
  }

  /**
    * Get the feature importance from the subtree
    *
    * This routine sums the importance from the children and adds the local
    * improvement to the feature used in this split
    *
    * @return feature importance as a vector
    */
  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    val ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex()) = ans(split.getIndex()) + improvement
    ans
  }

  /**
    * Compute Shapley feature attributions for a given input
    *
    * @param input for which to compute feature attributions.
    * @return array of Shapley feature attributions, one per input feature.
    */
  def shapley(input: Vector[AnyVal]): Array[Double] = {
    val importances = Array.fill[Double](input.length)(elem=0)
    shapleyRecurse(input, new FeaturePath(numFeatures), 1.0, 1.0, -1, importances)
    importances
  }

  /**
    * Get the Shapley feature attribution from a subtree
    *
    * @param input for which to compute feature attributions.
    * @param parentPath path of unique features arriving at parent node.
    * @param parentZeroFraction fraction of zero (cold) paths flowing to parent node.
    * @param parentOneFraction fraction of one (hot) paths flowing to parent node.
    * @param parentFeatureIndex index of feature on which the parent node splits.
    * @param importances array of feature attributions to modify.
    */
  def shapleyRecurse(
               input: Vector[AnyVal],
               parentPath: FeaturePath,
               parentZeroFraction: Double,
               parentOneFraction: Double,
               parentFeatureIndex: Int,
               importances: Array[Double]
             ): Unit = {
    var path = parentPath.copy().extend(parentZeroFraction, parentOneFraction, parentFeatureIndex)

    val (hot, hotTrain, cold, coldTrain) = if (this.split.turnLeft(input)) {
      (leftChild,leftTrain,rightChild,rightTrain)
    } else {
      (rightChild,rightTrain,leftChild,leftTrain)
    }
    var incomingZeroFraction = 1.0
    var incomingOneFraction = 1.0

    val k = path.path.take(path.length+1).indexWhere{x => x.featureIndex == split.getIndex() && x.featureIndex > -1}
    if (k > 0) {
      incomingZeroFraction = path.path(k).zeroFraction
      incomingOneFraction = path.path(k).oneFraction
      path = path.unwind(k)
    }
    var x = hot match {
      case node: RegressionTrainingNode => node.shapleyRecurse(
        input, path, incomingZeroFraction*hotTrain.size.toDouble/trainingData.size, incomingOneFraction, split.getIndex(), importances)
      case leaf: RegressionTrainingLeaf => leaf.shapleyRecurse(
        input, path, incomingZeroFraction*hotTrain.size.toDouble/trainingData.size, incomingOneFraction, split.getIndex(), importances)
    }
    x = cold match {
      case node: RegressionTrainingNode => node.shapleyRecurse(
        input, path, incomingZeroFraction*coldTrain.size.toDouble/trainingData.size, 0.0, split.getIndex(), importances)
      case leaf: RegressionTrainingLeaf => leaf.shapleyRecurse(
        input, path, incomingZeroFraction*coldTrain.size.toDouble/trainingData.size, 0.0, split.getIndex(), importances)
    }
  }
}

/**
  * Companion object to hold helper functions
  */
object RegressionTrainingNode {
  /**
    * Build a child node by pre-computing a split
    *
    * If there isn't a split, the child is a leaf; otherwise, the child is
    * another training node
    *
    * @param trainingData     for the child
    * @param leafLearner      to pass through
    * @param minLeafInstances minimum training instances per node
    * @param remainingDepth   the number of splits left
    * @param maxDepth         to compute depth via remainingDepth
    * @param numFeatures      to consider in the split
    * @return the child node, either a RegressionTrainingNode or TrainingLeaf
    */
  def buildChild(
                  trainingData: Seq[(Vector[AnyVal], Double, Double)],
                  leafLearner: Learner,
                  splitter: Splitter[Double],
                  minLeafInstances: Int,
                  remainingDepth: Int,
                  maxDepth: Int,
                  numFeatures: Int
                ): TrainingNode[AnyVal, Double] = {
    if (trainingData.size >= 2 * minLeafInstances && remainingDepth > 0 && trainingData.exists(_._2 != trainingData.head._2)) {
      val (leftSplit, leftDelta) = splitter.getBestSplit(trainingData, numFeatures, minLeafInstances)
      if (!leftSplit.isInstanceOf[NoSplit]) {
        new RegressionTrainingNode(trainingData, leafLearner, splitter, leftSplit, leftDelta, numFeatures, minLeafInstances, remainingDepth - 1, maxDepth)
      } else {
        new RegressionTrainingLeaf(trainingData, leafLearner, maxDepth - remainingDepth)
      }
    } else {
      new RegressionTrainingLeaf(trainingData, leafLearner, maxDepth - remainingDepth)
    }
  }
}
