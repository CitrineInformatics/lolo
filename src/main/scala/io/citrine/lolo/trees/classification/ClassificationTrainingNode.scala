package io.citrine.lolo.trees.classification

import io.citrine.lolo.trees.splits.{NoSplit, Split, Splitter}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.{Learner, TrainingRow}
import io.citrine.random.Random

import scala.collection.mutable

case class ClassificationTrainingNode(
    trainingData: Seq[TrainingRow[Char]],
    leftNode: TrainingNode[Char],
    rightNode: TrainingNode[Char],
    leafLearner: Learner[Char],
    split: Split,
    deltaImpurity: Double,
    numClasses: Int
) extends TrainingNode[Char] {

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def modelNode: ModelNode[Char] =
    InternalModelNode(
      split,
      leftNode.modelNode,
      rightNode.modelNode,
      numClasses,
      trainingData.map(_.weight).sum
    )

  override def featureImportance: mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    val ans = leftNode.featureImportance.zip(rightNode.featureImportance).map(p => p._1 + p._2)
    ans(split.index) = ans(split.index) + improvement
    ans
  }
}

object ClassificationTrainingNode {

  def build(
      trainingData: Seq[TrainingRow[Char]],
      leafLearner: Learner[Char],
      splitter: Splitter[Char],
      numFeatures: Int,
      minLeafInstances: Int,
      remainingDepth: Int,
      maxDepth: Int,
      numClasses: Int,
      rng: Random
  ): TrainingNode[Char] = {
    val sufficientData = trainingData.size >= 2 * minLeafInstances &&
      remainingDepth > 0 &&
      trainingData.exists(_.label != trainingData.head.label)
    val (split: Split, deltaImpurity: Double) = if (sufficientData) {
      splitter.getBestSplit(trainingData, numFeatures, minLeafInstances, rng)
    } else {
      (NoSplit(), 0.0)
    }
    split match {
      case _: NoSplit =>
        ClassificationTrainingLeaf.build(trainingData, leafLearner, maxDepth - remainingDepth, rng)
      case split: Split =>
        val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
        val leftNode = ClassificationTrainingNode.build(
          trainingData = leftTrain,
          leafLearner = leafLearner,
          splitter = splitter,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          remainingDepth = remainingDepth - 1,
          maxDepth = maxDepth,
          numClasses = numClasses,
          rng = rng
        )
        val rightNode = ClassificationTrainingNode.build(
          trainingData = rightTrain,
          leafLearner = leafLearner,
          splitter = splitter,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          remainingDepth = remainingDepth - 1,
          maxDepth = maxDepth,
          numClasses = numClasses,
          rng = rng
        )
        ClassificationTrainingNode(
          trainingData = trainingData,
          leftNode = leftNode,
          rightNode = rightNode,
          leafLearner = leafLearner,
          split = split,
          deltaImpurity = deltaImpurity,
          numClasses = numClasses
        )
    }
  }
}
