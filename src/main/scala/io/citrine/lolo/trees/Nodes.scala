package io.citrine.lolo.trees

import io.citrine.lolo.trees.splits.Split

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
  def getNode(): ModelNode[T, S]

  /**
    * Get the feature importance of the subtree below this node
    *
    * @return feature importance as a vector
    */
  def getFeatureImportance(): Array[Double]
}

/**
  * Trait to hold prediction interface
  *
  * @tparam T type of the input vector
  * @tparam S type of the model output
  */
abstract trait ModelNode[T <: AnyVal, S] extends Serializable {
  def predict(input: Vector[T]): S
}

/**
  * Internal node in the decision tree
  *
  * @param split to decide which branch to take
  * @param left  branch node
  * @param right branch node
  * @tparam T type of the input
  * @tparam S type of the output
  */
class InternalModelNode[T <: AnyVal, S](
                                         split: Split,
                                         left: ModelNode[T, S],
                                         right: ModelNode[T, S]
                                       ) extends ModelNode[T, S] {
  /**
    * Just propagate the prediction call through the appropriate child
    *
    * @param input to predict for
    * @return prediction
    */
  override def predict(input: Vector[T]): S = {
    if (split.turnLeft(input)) {
      left.predict(input)
    } else {
      right.predict(input)
    }
  }
}
