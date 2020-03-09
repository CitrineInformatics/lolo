package io.citrine.lolo.trees.regression

import breeze.linalg.DenseVector
import breeze.numerics.abs
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
    new InternalModelNode[PredictionResult[Double]](split, leftChild.getNode(), rightChild.getNode(), numFeatures, 1, trainingData.size.toDouble)
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
    * Get mean absolute Shapley values across training data
    *
    * @return vector of mean absolute Shapley values
    *         One DenseVector[Double] per feature, each of length equal to the output dimension.
    */
  override def getShapley(): Option[Vector[DenseVector[Double]]] = {
    val shaps = trainingData.map{d=>getNode().shapley(d._1)}
    if (!shaps.head.isDefined) {
      None
    }
    assert(shaps.forall(x=>x.isDefined))
    def sumReducer(a: Option[Vector[DenseVector[Double]]],
                   b: Option[Vector[DenseVector[Double]]]): Option[Vector[DenseVector[Double]]] = {
      (a ++ b).reduceOption[Vector[DenseVector[Double]]]{case (x,y) => x.zip(y).map{case (v1,v2) => (abs(v1) + abs(v2))}}
    }
    val scale = 1.0/shaps.length
    shaps.reduce(sumReducer).map{x=>x.map{y=>scale*y}}
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
