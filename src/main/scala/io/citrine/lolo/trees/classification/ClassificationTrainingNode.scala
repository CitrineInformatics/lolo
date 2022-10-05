package io.citrine.lolo.trees.classification

import io.citrine.lolo.trees.splits.{NoSplit, Split, Splitter}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.{Learner, PredictionResult}
import io.citrine.random.Random

case class ClassificationTrainingNode(
    trainingData: Seq[(Vector[AnyVal], Char, Double)],
    leftNode: TrainingNode[Char],
    rightNode: TrainingNode[Char],
    leafLearner: Learner,
    split: Split,
    deltaImpurity: Double,
    numClasses: Int
) extends TrainingNode[Char] {

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getModelNode(): ModelNode[PredictionResult[Char]] =
    InternalModelNode(
      split,
      leftNode.getModelNode(),
      rightNode.getModelNode(),
      numClasses,
      trainingData.map(_._3).sum
    )

  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    val ans = leftNode.getFeatureImportance().zip(rightNode.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.index) = ans(split.index) + improvement
    ans
  }
}

object ClassificationTrainingNode {

  def build(
      trainingData: Seq[(Vector[AnyVal], Char, Double)],
      leafLearner: Learner,
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
      trainingData.exists(_._2 != trainingData.head._2)
    if (sufficientData) {
      val (split: Split, deltaImpurity: Double) =
        splitter.getBestSplit(trainingData, numFeatures, minLeafInstances, rng)
      if (!split.isInstanceOf[NoSplit]) {
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
      } else {
        ClassificationTrainingLeaf.build(trainingData, leafLearner, maxDepth - remainingDepth, rng)
      }
    } else {
      ClassificationTrainingLeaf.build(trainingData, leafLearner, maxDepth - remainingDepth, rng)
    }
  }
}
