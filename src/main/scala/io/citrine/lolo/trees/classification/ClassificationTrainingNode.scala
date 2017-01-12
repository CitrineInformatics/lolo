package io.citrine.lolo.trees.classification

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.trees.splits.{ClassificationSplitter, NoSplit, Split}

/**
  * Created by maxhutch on 1/12/17.
  */
class ClassificationTrainingNode(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                                  split: Split,
                                  deltaImpurity: Double,
                                  numFeatures: Int,
                                  remainingDepth: Int,
                                  maxDepth: Int
                                ) extends TrainingNode(trainingData, remainingDepth) {

  assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
  assert(!split.isInstanceOf[NoSplit], s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}")
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0 && leftTrain.exists(_._2 != leftTrain.head._2)) {
    lazy val (leftSplit, leftDelta) = ClassificationSplitter.getBestSplit(leftTrain, numFeatures)
    if (!leftSplit.isInstanceOf[NoSplit]) {
      new ClassificationTrainingNode(leftTrain, leftSplit, leftDelta, numFeatures, remainingDepth - 1, maxDepth)
    } else {
      new ClassificationTrainingLeaf(leftTrain, maxDepth - remainingDepth)
    }
  } else {
    new ClassificationTrainingLeaf(leftTrain, maxDepth - remainingDepth)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0 && rightTrain.exists(_._2 != rightTrain.head._2)) {
    lazy val (rightSplit, rightDelta) = ClassificationSplitter.getBestSplit(rightTrain, numFeatures)
    if (!rightSplit.isInstanceOf[NoSplit]) {
      new ClassificationTrainingNode(rightTrain, rightSplit, rightDelta, numFeatures, remainingDepth - 1, maxDepth)
    } else {
      new ClassificationTrainingLeaf(rightTrain, maxDepth - remainingDepth)
    }
  } else {
    new ClassificationTrainingLeaf(rightTrain, maxDepth - remainingDepth)
  }

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Char]] = new InternalModelNode(
    split, leftChild.getNode(), rightChild.getNode()
  )

  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex) = ans(split.getIndex) + improvement
    ans
  }
}
