package io.citrine.lolo.trees

import io.citrine.lolo.Model
import io.citrine.lolo.results.PredictionResult
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
  def getNode(): Model[PredictionResult[S]]

  /**
    * Get the feature importance of the subtree below this node
    *
    * @return feature importance as a vector
    */
  def getFeatureImportance(): Array[Double]
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
                                         left: Model[T],
                                         right: Model[T]
                                       ) extends Model[T] {
  /**
    * Just propagate the prediction call through the appropriate child
    *
    * @param input to predict for
    * @return prediction
    */
  override def transform(input: Seq[Vector[Any]]): T = {
    if (split.turnLeft(input.head.asInstanceOf[Vector[AnyVal]])) {
      left.transform(input)
    } else {
      right.transform(input)
    }
  }
}
