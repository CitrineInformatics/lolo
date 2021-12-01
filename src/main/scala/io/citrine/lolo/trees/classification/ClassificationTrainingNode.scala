package io.citrine.lolo.trees.classification

import io.citrine.lolo.trees.splits.{NoSplit, Split, Splitter}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingLeaf, TrainingNode}
import io.citrine.lolo.{Learner, PredictionResult}

/**
  * Created by maxhutch on 1/12/17.
  *
  * @param trainingData     that this node sees
  * @param leafLearner      to train on leaves
  * @param split            used to split this node's direct descendants
  * @param deltaImpurity    change in training set impurity between this node vs direct descendants
  * @param numFeatures      number of features used in selecting splits
  * @param minLeafInstances minimum training instances per node
  * @param remainingDepth   to stop growing the node
  * @param maxDepth         maximum depth of tree
  * @param numClasses       total number of classes, used as an output dimensionality
  * @param splitter         used to select splits
  */
class ClassificationTrainingNode(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                                  leafLearner: Learner,
                                  split: Split,
                                  deltaImpurity: Double,
                                  numFeatures: Int,
                                  minLeafInstances: Int,
                                  remainingDepth: Int,
                                  maxDepth: Int,
                                  numClasses: Int,
                                  splitter: Splitter[Char]
                                ) extends TrainingNode(trainingData, remainingDepth) {

  assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
  assert(!split.isInstanceOf[NoSplit], s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}")

  lazy val leftChild = ClassificationTrainingNode.buildChild(leftTrain, leafLearner, minLeafInstances, remainingDepth, maxDepth, numFeatures, numClasses, splitter)

  lazy val rightChild = ClassificationTrainingNode.buildChild(rightTrain, leafLearner, minLeafInstances, remainingDepth, maxDepth, numFeatures, numClasses, splitter)


  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Char]] = new InternalModelNode(
    split, leftChild.getNode(), rightChild.getNode(), numClasses, trainingData.size.toDouble
  )

  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    val ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex()) = ans(split.getIndex()) + improvement
    ans
  }
}

object ClassificationTrainingNode {
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
    * @param numClasses       total number of classes, used as an output dimensionality
    * @param splitter         used to select splits
    * @return the child node, either a RegressionTrainingNode or TrainingLeaf
    */
  def buildChild(
                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                  leafLearner: Learner,
                  minLeafInstances: Int,
                  remainingDepth: Int,
                  maxDepth: Int,
                  numFeatures: Int,
                  numClasses: Int,
                  splitter: Splitter[Char]
                ): TrainingNode[AnyVal, Char] = {
    if (trainingData.size >= 2 * minLeafInstances && remainingDepth > 0 && trainingData.exists(_._2 != trainingData.head._2)) {
      val (leftSplit, leftDelta) = splitter.getBestSplit(trainingData, numFeatures, minLeafInstances)
      if (!leftSplit.isInstanceOf[NoSplit]) {
        new ClassificationTrainingNode(trainingData, leafLearner, leftSplit, leftDelta, numFeatures, minLeafInstances, remainingDepth - 1, maxDepth, numClasses, splitter)
      } else {
        new TrainingLeaf(trainingData, leafLearner, maxDepth - remainingDepth)
      }
    } else {
      new TrainingLeaf(trainingData, leafLearner, maxDepth - remainingDepth)
    }
  }
}
