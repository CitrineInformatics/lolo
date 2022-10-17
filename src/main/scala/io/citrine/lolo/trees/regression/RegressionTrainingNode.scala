package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.splits.{NoSplit, Split, Splitter}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.Learner
import io.citrine.random.Random

case class RegressionTrainingNode(
    trainingData: Seq[(Vector[AnyVal], Double, Double)],
    leftNode: TrainingNode[Double],
    rightNode: TrainingNode[Double],
    leafLearner: Learner[Double],
    split: Split,
    deltaImpurity: Double
) extends TrainingNode[Double] {

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def modelNode: ModelNode[Double] = {
    InternalModelNode(
      split,
      leftNode.modelNode,
      rightNode.modelNode,
      outputDimension = 1,
      trainingData.map(_._3).sum
    )
  }

  /**
    * Get the feature importance from the subtree
    *
    * This routine sums the importance from the children and adds the local
    * improvement to the feature used in this split
    *
    * @return feature importance as a vector
    */
  override def featureImportance: scala.collection.mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    val ans = leftNode.featureImportance.zip(rightNode.featureImportance).map(p => p._1 + p._2)
    ans(split.index) = ans(split.index) + improvement
    ans
  }
}

object RegressionTrainingNode {

  /**
    * Build a node by computing the split and recursively building the child nodes if they exist.
    * If there isn't a split, this becomes a leaf node.
    *
    * @param trainingData     on which to split
    * @param leafLearner      to pass through until we get to a leaf node
    * @param splitter         determines the best split
    * @param minLeafInstances minimum training instances per node
    * @param remainingDepth   the maximum number of splits left
    * @param maxDepth         to compute depth via remainingDepth
    * @param numFeatures      to consider in the split
    * @param rng              random number generator, for reproducibility
    * @return the child node, either a RegressionTrainingNode or TrainingLeaf
    */
  def build(
      trainingData: Seq[(Vector[AnyVal], Double, Double)],
      leafLearner: Learner[Double],
      splitter: Splitter[Double],
      numFeatures: Int,
      minLeafInstances: Int,
      remainingDepth: Int,
      maxDepth: Int,
      rng: Random
  ): TrainingNode[Double] = {
    val sufficientData = trainingData.size >= 2 * minLeafInstances &&
      remainingDepth > 0 &&
      trainingData.exists(_._2 != trainingData.head._2)
    val (split: Split, deltaImpurity: Double) = if (sufficientData) {
      splitter.getBestSplit(trainingData, numFeatures, minLeafInstances, rng)
    } else {
      (NoSplit(), 0.0)
    }
    split match {
      case _: NoSplit =>
        RegressionTrainingLeaf.build(trainingData, leafLearner, maxDepth - remainingDepth, rng)
      case split: Split =>
        val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
        val leftNode = RegressionTrainingNode.build(
          trainingData = leftTrain,
          leafLearner = leafLearner,
          splitter = splitter,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          remainingDepth = remainingDepth - 1,
          maxDepth = maxDepth,
          rng = rng
        )
        val rightNode = RegressionTrainingNode.build(
          trainingData = rightTrain,
          leafLearner = leafLearner,
          splitter = splitter,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          remainingDepth = remainingDepth - 1,
          maxDepth = maxDepth,
          rng = rng
        )
        RegressionTrainingNode(
          trainingData = trainingData,
          leftNode = leftNode,
          rightNode = rightNode,
          leafLearner = leafLearner,
          split = split,
          deltaImpurity = deltaImpurity
        )
    }
  }
}
