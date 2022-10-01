package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.splits.{NoSplit, Split, Splitter}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.{Learner, PredictionResult}
import io.citrine.random.Random

case class RegressionTrainingNode(
    trainingData: Seq[(Vector[AnyVal], Double, Double)],
    leftNode: TrainingNode[AnyVal, Double],
    rightNode: TrainingNode[AnyVal, Double],
    leafLearner: Learner,
    split: Split,
    deltaImpurity: Double
) extends TrainingNode[AnyVal, Double] {

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getModelNode(): ModelNode[PredictionResult[Double]] = {
    new InternalModelNode[PredictionResult[Double]](
      split,
      leftNode.getModelNode(),
      rightNode.getModelNode(),
      1,
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
  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    val ans = leftNode.getFeatureImportance().zip(rightNode.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex) = ans(split.getIndex) + improvement
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
    * @param minLeafInstances minimum training instances per node
    * @param remainingDepth   the maximum number of splits left
    * @param maxDepth         to compute depth via remainingDepth
    * @param numFeatures      to consider in the split
    * @param rng              random number generator, for reproducibility
    * @return the child node, either a RegressionTrainingNode or TrainingLeaf
    */
  def build(
      trainingData: Seq[(Vector[AnyVal], Double, Double)],
      leafLearner: Learner,
      splitter: Splitter[Double],
      numFeatures: Int,
      minLeafInstances: Int,
      remainingDepth: Int,
      maxDepth: Int,
      rng: Random
  ): TrainingNode[AnyVal, Double] = {
    val sufficientData = trainingData.size >= 2 * minLeafInstances &&
      remainingDepth > 0 &&
      trainingData.exists(_._2 != trainingData.head._2)
    if (sufficientData) {
      val (split: Split, deltaImpurity: Double) =
        splitter.getBestSplit(trainingData, numFeatures, minLeafInstances, rng)
      if (!split.isInstanceOf[NoSplit]) {
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
      } else {
        RegressionTrainingLeaf.build(trainingData, leafLearner, maxDepth - remainingDepth, rng)
      }
    } else {
      RegressionTrainingLeaf.build(trainingData, leafLearner, maxDepth - remainingDepth, rng)
    }
  }
}
