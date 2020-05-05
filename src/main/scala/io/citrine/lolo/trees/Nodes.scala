package io.citrine.lolo.trees

import breeze.linalg.DenseMatrix
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

  /**
    * Compute Shapley feature attributions for a given input in this node's subtree
    *
    * @param input for which to compute feature attributions.
    * @return array of vector-valued attributions for each feature
    *         One Vector[Double] per feature, each of length equal to the output dimension.
    */
  def shapley(input: Vector[AnyVal], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]]

  /**
    * Get the Shapley feature attribution from a subtree
    *
    * This uses the TreeShap algorithm given in https://arxiv.org/abs/1802.03888
    *
    * @param input for which to compute feature attributions.
    * @param parentPath path of unique features arriving at parent node.
    * @param parentZeroFraction fraction of zero (cold) paths flowing to parent node.
    * @param parentOneFraction fraction of one (hot) paths flowing to parent node.
    * @param parentFeatureIndex index of feature on which the parent node splits.
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  private[lolo] def shapleyRecurse(
                                    input: Vector[AnyVal],
                                    omitFeatures: Set[Int] = Set(),
                                    parentPath: DecisionPath,
                                    parentZeroFraction: Double,
                                    parentOneFraction: Double,
                                    parentFeatureIndex: Int
                    ): DenseMatrix[Double]

  /**
    * Weight of training data in subtree, specifically the number of data for unweighted training sets
    *
    * @return total weight of training weight in subtree
    */
  private[lolo] def getTrainingWeight(): Double
}

/**
  * Internal node in the decision tree
  *
  * @param split to decide which branch to take
  * @param left  branch node
  * @param right branch node
  * @param numFeatures number of features, used for Shapley computation
  * @param outputDimension dimension of model output, used for Shapley computation
  *                        1 for single-task regression, or equal to the number of classification categories.
  * @param trainingWeight weight of training data in subtree (i.e. size of unweighted training set)
  * @tparam T type of the output
  */
class InternalModelNode[T <: PredictionResult[Any]](
                                                     split: Split,
                                                     left: ModelNode[T],
                                                     right: ModelNode[T],
                                                     numFeatures: Int,
                                                     outputDimension: Int,
                                                     trainingWeight: Double
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

  /**
    * Compute Shapley feature attributions for a given input
    *
    * @param input for which to compute feature attributions.
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  override def shapley(input: Vector[AnyVal], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
    // the featureIndex = -1 signals that this is the base case
    Some(shapleyRecurse(input, omitFeatures, new DecisionPath(input.length), 1.0, 1.0, -1))
  }

  /**
    * Get the Shapley feature attribution from a subtree
    *
    * @param input for which to compute feature attributions.
    * @param parentPath path of unique features arriving at parent node.
    * @param parentZeroFraction fraction of zero (cold) paths flowing to parent node.
    * @param parentOneFraction fraction of one (hot) paths flowing to parent node.
    * @param parentFeatureIndex index of feature on which the parent node splits.
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  def shapleyRecurse(
                      input: Vector[AnyVal],
                      omitFeatures: Set[Int],
                      parentPath: DecisionPath,
                      parentZeroFraction: Double,
                      parentOneFraction: Double,
                      parentFeatureIndex: Int
                    ): DenseMatrix[Double] = {
    val (hot, cold) = if (this.split.turnLeft(input)) {
      (left,right)
    } else {
      (right,left)
    }

    if (omitFeatures.contains(split.getIndex())) {
      val hotPortion = hot.getTrainingWeight() / trainingWeight
      val coldPortion = cold.getTrainingWeight() / trainingWeight
      val hotContrib = hot match {  // Traverse one subtree.
        case _: ModelNode[T] | _: ModelLeaf[T] => hot.shapleyRecurse(
          input, omitFeatures, parentPath, parentZeroFraction, parentOneFraction, parentFeatureIndex)
        case _ => throw new RuntimeException("Tree children must be of type ModelNode[T] or ModelLeaf[T].")
      }
      val coldContrib = cold match {  // Traverse the other subtree.
        case _: ModelNode[T] | _: ModelLeaf[T] => cold.shapleyRecurse(
          input, omitFeatures, parentPath, parentZeroFraction, parentOneFraction, parentFeatureIndex)
        case _ => throw new RuntimeException("Tree children must be of type ModelNode[T] or ModelLeaf[T].")
      }
      hotPortion * hotContrib + coldPortion * coldContrib
    } else {
      var path = parentPath.copy().extend(parentZeroFraction, parentOneFraction, parentFeatureIndex)

      // If this node in the tree splits on a feature that is already present in the feature path, unwind that feature from the path to prevent duplication.
      val previousNode = path.features.find(_.featureIndex == split.getIndex())
      val (incomingZeroFraction: Double, incomingOneFraction: Double) = previousNode match {
        case Some(node) =>
          path = path.unwind(node.featureIndex)
          (
            node.weightWhenExcluded,  // Proportion of zero paths for this feature that flow down to this branch.
            node.weightWhenIncluded    // Proportion of one paths for this feature that flow down to this branch.
          )
        case None =>
          // This is the first split on this feature in the present branch's ancestry, so all of the zero and one paths flow down to it.
          (1.0, 1.0)
      }

      val hotContrib = hot match {  // Traverse one subtree.
        case _: ModelNode[T] | _: ModelLeaf[T] => hot.shapleyRecurse(
          input, omitFeatures, path, incomingZeroFraction * hot.getTrainingWeight() / trainingWeight, incomingOneFraction, split.getIndex())
        case _ => throw new RuntimeException("Tree children must be of type ModelNode[T] or ModelLeaf[T].")
      }

      val coldContrib = cold match {  // Traverse the other subtree.
        case _: ModelNode[T] | _: ModelLeaf[T] => cold.shapleyRecurse(
          input, omitFeatures, path, incomingZeroFraction * cold.getTrainingWeight() / trainingWeight, 0.0, split.getIndex())
        case _ => throw new RuntimeException("Tree children must be of type ModelNode[T] or ModelLeaf[T].")
      }

      coldContrib + hotContrib
    }
  }

  override def getTrainingWeight(): Double = trainingWeight
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
    new ModelLeaf(leafLearner.train(trainingData).getModel().asInstanceOf[Model[PredictionResult[T]]], depth, trainingData.head._1.length, trainingData.size.toDouble)
  }

  override def getFeatureImportance(): mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
}

class ModelLeaf[T](model: Model[PredictionResult[T]], depth: Int, numFeatures: Int, trainingWeight: Double) extends ModelNode[PredictionResult[T]] {
  override def transform(input: Vector[AnyVal]): (PredictionResult[T], TreeMeta) = {
    (model.transform(Seq(input)), TreeMeta(depth))
  }

  /**
    * Get the Shapley feature attribution from a subtree
    *
    * @param input for which to compute feature attributions.
    * @param parentPath path of unique features arriving at parent node.
    * @param parentZeroFraction fraction of zero (cold) paths flowing to parent node.
    * @param parentOneFraction fraction of one (hot) paths flowing to parent node.
    * @param parentFeatureIndex index of feature on which the parent node splits.
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  def shapleyRecurse(
                      input: Vector[AnyVal],
                      omitFeatures: Set[Int],
                      parentPath: DecisionPath,
                      parentZeroFraction: Double,
                      parentOneFraction: Double,
                      parentFeatureIndex: Int
                    ): DenseMatrix[Double] = {

    // First, account for the split that led into this leaf
    val path = if (!omitFeatures.contains(parentFeatureIndex)) {
      parentPath.copy().extend(parentZeroFraction, parentOneFraction, parentFeatureIndex)
    } else {
      parentPath.copy()
    }

    // For each feature in the decision path, unwind that feature to remove its impact on the combinatorial factors
    // and then compute its contribution to the shapley value of that feature as:
    // (difference in the weights when included and excluded) * (weight and combinatorial factor from other features) * predicted value

    val out = DenseMatrix.zeros[Double](1, input.length)
    path.features.foreach { node =>
      val w = path.unwind(node.featureIndex).totalWeight
      this.model.transform(Seq(input)).getExpected().head match {
        case v: Double =>
          out(0, node.featureIndex) = w * (node.weightWhenIncluded - node.weightWhenExcluded) * v
        case _ => throw new NotImplementedError()
      }
    }
    out
  }

  override def getTrainingWeight(): Double = trainingWeight

  override def shapley(input: Vector[AnyVal], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = None
}
