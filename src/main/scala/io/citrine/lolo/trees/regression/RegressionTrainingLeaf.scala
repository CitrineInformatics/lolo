package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.{ModelNode, TrainingNode}
import io.citrine.lolo.{Learner, Model, PredictionResult}

import scala.collection.mutable

/**
  * Average the training data to make a leaf prediction
  *
  * @param trainingData to train on
  */
class RegressionTrainingLeaf(
                              trainingData: Seq[(Vector[AnyVal], Double, Double)],
                              leafLearner: Learner,
                              depth: Int
                            ) extends TrainingNode(
  trainingData = trainingData,
  remainingDepth = 0
) {
  /**
    * Average the training data
    *
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[PredictionResult[Double]] = {
    new RegressionLeaf(leafLearner.train(trainingData).getModel().asInstanceOf[Model[PredictionResult[Double]]], depth)
  }

  override def getFeatureImportance(): mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
}
