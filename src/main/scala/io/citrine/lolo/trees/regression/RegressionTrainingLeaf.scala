package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.{ModelLeaf, ModelNode, TrainingNode}
import io.citrine.lolo.{Learner, Model, PredictionResult}

import scala.collection.mutable

/**
  * Training leaf node for regression trees
  * Created by maxhutch on 3/8/17.
  */
class RegressionTrainingLeaf(
                              trainingData: Seq[(Vector[AnyVal], Double, Double)],
                              leafLearner: Learner,
                              depth: Int
                            ) extends TrainingNode(trainingData, depth) {

  /**
    * Wrap the leaf model (previously trained) in a lightweight leaf node
    *
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[PredictionResult[Double]] = {
    new ModelLeaf(model.asInstanceOf[Model[PredictionResult[Double]]], depth)
  }

  /**
    * Pull the leaf model's feature importance and rescale it by the remaining impurity
    * @return feature importance as a vector
    */
  def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = {
    importance match {
      case Some(x) =>
        val expectations: (Double, Double) = trainingData.map{ case (v, l, w) =>
          (l * w, l * l * w)
        }.reduce((u: (Double, Double), v: (Double, Double)) => (u._1 + v._1, u._2 + v._2))
        val impurity = expectations._2 / trainingData.size - Math.pow(expectations._1 / trainingData.size, 2.0)
        mutable.ArraySeq(x: _*).map(_ * impurity)
      case None => mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
    }
  }

  /** Train the leaf learner on the training data */
  val leafTrainingResult = leafLearner.train(trainingData)
  /** Pull out the model for future use */
  val model = leafTrainingResult.getModel().asInstanceOf[Model[PredictionResult[Double]]]
  /** Pull out the importance for future use */
  val importance = leafTrainingResult.getFeatureImportance()
}