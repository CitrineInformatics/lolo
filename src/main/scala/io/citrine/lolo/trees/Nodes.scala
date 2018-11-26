package io.citrine.lolo.trees

import io.citrine.lolo.trees.splits.Split
import io.citrine.lolo.{Learner, Model, PredictionResult}

import scala.collection.mutable

/**
  * Class to provide getNode interface for internal and leaf training nodes
  * Created by maxhutch on 11/29/16.
  *
  * @param trainingData   that this node sees
  * @param remainingDepth to stop growing the node
  * @tparam T type of the input vector
  * @tparam S type of the model output
  */
abstract class TrainingNode[T <: AnyVal, S](
                                             trainingData: Seq[(Vector[T], S, Double)],
                                             remainingDepth: Int = Int.MaxValue
                                           ) extends Serializable {
  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[PredictionResult[S]]

  /**
    * Get the feature importance of the subtree below this node
    *
    * @return feature importance as a vector
    */
  def getFeatureImportance(): mutable.ArraySeq[Double]
}

trait ModelNode[T <: PredictionResult[Any]] extends Serializable {
  def transform(input: Vector[AnyVal]): (T, TreeMeta)
}

/**
  * Internal node in the decision tree
  *
  * @param split to decide which branch to take
  * @param left  branch node
  * @param right branch node
  * @tparam T type of the output
  */
class InternalModelNode[T <: PredictionResult[Any]](
                                                     split: Split,
                                                     left: ModelNode[T],
                                                     right: ModelNode[T]
                                                   ) extends ModelNode[T] {
  /**
    * Just propagate the prediction call through the appropriate child
    *
    * @param input to predict for
    * @return prediction
    */
  override def transform(input: Vector[AnyVal]): (T, TreeMeta) = {
    if (split.turnLeft(input)) {
      left.transform(input)
    } else {
      right.transform(input)
    }
  }
}

/**
  * Average the training data to make a leaf prediction
  *
  * @param trainingData to train on
  */
class TrainingLeaf[T](
                       trainingData: Seq[(Vector[AnyVal], T, Double)],
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
  def getNode(): ModelNode[PredictionResult[T]] = {
    new ModelLeaf(leafLearner.train(trainingData).getModel().asInstanceOf[Model[PredictionResult[T]]], depth)
  }

  override def getFeatureImportance(): mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
}

class ModelLeaf[T](model: Model[PredictionResult[T]], depth: Int) extends ModelNode[PredictionResult[T]] {
  override def transform(input: Vector[AnyVal]): (PredictionResult[T], TreeMeta) = {
    (model.transform(Seq(input)), TreeMeta(depth))
  }
}
