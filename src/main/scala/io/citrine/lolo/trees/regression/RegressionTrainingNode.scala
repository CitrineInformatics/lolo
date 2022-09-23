package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.splits.{NoSplit, Split, Splitter}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.{Learner, PredictionResult}
import io.citrine.random.Random

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
    maxDepth: Int,
    rng: Random
) extends TrainingNode(
      trainingData = trainingData,
      remainingDepth = remainingDepth
    ) {

  // TODO (PLA-10388): see if there's a way to get the rng for this (and the other nodes) out of the constructor
  // val (split: Split, deltaImpurity: Double) = RegressionSplitter.getBestSplit(trainingData, numFeatures)
  assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
  assert(
    !split.isInstanceOf[NoSplit],
    s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}"
  )
  private val leftRng = rng.split()
  private val rightRng = rng.split()

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  assert(
    leftTrain.nonEmpty && rightTrain.nonEmpty,
    s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}"
  )

  lazy val leftChild = RegressionTrainingNode.buildChild(
    leftTrain,
    leafLearner,
    splitter,
    minLeafInstances,
    remainingDepth,
    maxDepth,
    numFeatures,
    leftRng
  )
  lazy val rightChild = RegressionTrainingNode.buildChild(
    rightTrain,
    leafLearner,
    splitter,
    minLeafInstances,
    remainingDepth,
    maxDepth,
    numFeatures,
    rightRng
  )

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Double]] = {
    new InternalModelNode[PredictionResult[Double]](
      split,
      leftChild.getNode(),
      rightChild.getNode(),
      1,
      trainingData.size.toDouble
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
    val ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex()) = ans(split.getIndex()) + improvement
    ans
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
    * @param rng              random number generator for reproducibility
    * @return the child node, either a RegressionTrainingNode or TrainingLeaf
    */
  def buildChild(
      trainingData: Seq[(Vector[AnyVal], Double, Double)],
      leafLearner: Learner,
      splitter: Splitter[Double],
      minLeafInstances: Int,
      remainingDepth: Int,
      maxDepth: Int,
      numFeatures: Int,
      rng: Random
  ): TrainingNode[AnyVal, Double] = {
    if (
      trainingData.size >= 2 * minLeafInstances && remainingDepth > 0 && trainingData.exists(
        _._2 != trainingData.head._2
      )
    ) {
      val (leftSplit, leftDelta) = splitter.getBestSplit(trainingData, numFeatures, minLeafInstances)
      if (!leftSplit.isInstanceOf[NoSplit]) {
        new RegressionTrainingNode(
          trainingData,
          leafLearner,
          splitter,
          leftSplit,
          leftDelta,
          numFeatures,
          minLeafInstances,
          remainingDepth - 1,
          maxDepth,
          rng
        )
      } else {
        new RegressionTrainingLeaf(trainingData, leafLearner, maxDepth - remainingDepth, rng)
      }
    } else {
      new RegressionTrainingLeaf(trainingData, leafLearner, maxDepth - remainingDepth, rng)
    }
  }
}
