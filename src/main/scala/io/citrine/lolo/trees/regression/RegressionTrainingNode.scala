package io.citrine.lolo.trees.regression

import io.citrine.lolo.{Learner, PredictionResult}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode}
import io.citrine.lolo.trees.splits.{NoSplit, Split}

/**
  * Created by maxhutch on 1/12/17.
  */
  class RegressionTrainingNode(
                                trainingData: Seq[(Vector[AnyVal], Double, Double)],
                                leafLearner: Learner,
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

    lazy val leftChild = RegressionTree.buildChild(leftTrain, leafLearner, minLeafInstances, remainingDepth, maxDepth, numFeatures)

    lazy val rightChild = RegressionTree.buildChild(rightTrain, leafLearner, minLeafInstances, remainingDepth, maxDepth, numFeatures)

    /**
      * Get the lightweight prediction node for the output tree
      *
      * @return lightweight prediction node
      */
    override def getNode(): ModelNode[PredictionResult[Double]] = {
      new InternalModelNode[PredictionResult[Double]](split, leftChild.getNode(), rightChild.getNode())
    }

    override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
      val improvement = deltaImpurity
      var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
      ans(split.getIndex) = ans(split.getIndex) + improvement
      ans
    }
  }
