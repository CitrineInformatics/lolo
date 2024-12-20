package io.citrine.lolo.trees

import breeze.linalg.DenseMatrix
import io.citrine.lolo.api.{Model, PredictionResult}
import io.citrine.lolo.trees.splits.Split
import breeze.linalg.sum

trait ModelNode[+T] extends Serializable {

  /** Transform the inputs, returning a prediction result and meta information regarding the tree depth. */
  def transform(input: Vector[AnyVal]): (PredictionResult[T], TreeMeta)

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
    * This uses an algorithm based on TreeShap from https://arxiv.org/abs/1802.03888
    *
    * The algorithm recursively tracks the feature that is used for each split and the share of the training data
    * that went to the left and the right of the split.  In the leaves, these features and their weights are used
    * to construct the leaf's contribution to the SHAP values, which are then summed.
    *
    * @param input          for which to compute feature attributions.
    * @param featureWeights Map from feature index to [[FeatureWeightFactor]], which stores the weight of the child
    *                       of the split when the feature is known vs unknown
    * @return matrix of attributions for each feature and output
    *         One row per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  private[lolo] def shapleyRecurse(
      input: Vector[AnyVal],
      omitFeatures: Set[Int] = Set(),
      featureWeights: Map[Int, FeatureWeightFactor]
  ): DenseMatrix[Double]

  /**
    * Weight of training data in subtree, specifically the number of data for unweighted training sets
    *
    * @return total weight of training weight in subtree
    */
  private[lolo] def trainingWeight: Double
}

/**
  * Internal node in the decision tree
  *
  * @param split           to decide which branch to take
  * @param left            branch node
  * @param right           branch node
  * @param outputDimension dimension of model output, used for Shapley computation
  *                        1 for single-task regression, or equal to the number of classification categories.
  * @param trainingWeight  weight of training data in subtree (i.e. size of unweighted training set)
  * @tparam T type of the output
  */
case class InternalModelNode[+T](
    split: Split,
    left: ModelNode[T],
    right: ModelNode[T],
    outputDimension: Int,
    trainingWeight: Double
) extends ModelNode[T] {

  @transient private lazy val leftPortion = left.trainingWeight / trainingWeight
  @transient private lazy val rightPortion = right.trainingWeight / trainingWeight

  /**
    * Just propagate the prediction call through the appropriate child
    *
    * @param input to predict for
    * @return prediction
    */
  override def transform(input: Vector[AnyVal]): (PredictionResult[T], TreeMeta) = {
    if (split.turnLeft(input)) {
      left.transform(input)
    } else {
      right.transform(input)
    }
  }

  override def shapley(input: Vector[AnyVal], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
    // Kick off a recursive procedure, the base case of which is an empty map
    Some(shapleyRecurse(input, omitFeatures, Map()))
  }

  /**
    * On the way down: append this node to the set of features that have been encountered in this path
    * through the decision tree unless this feature is in the omitted features list.
    *
    * On the way up: sum the contributions from the two children of this node.  If the feature that this node splits
    * on is in the omitted features list, then multiply the contributions by the share of the training data that
    * went to each child in lieu of including the feature on the way down.
    */
  def shapleyRecurse(
      input: Vector[AnyVal],
      omitFeatures: Set[Int],
      featureWeights: Map[Int, FeatureWeightFactor]
  ): DenseMatrix[Double] = {
    val featureIndex = split.index
    val goLeft = split.turnLeft(input)
    // The hot node is the one that is selected when the feature is present
    // If the feature is omitted or unknown, then the weight assigned to each child is taken
    // as the share of the training data that got split into each child
    val (hot, cold) = if (goLeft) (left, right) else (right, left)
    val (hotPortion, coldPortion) = if (goLeft) (leftPortion, rightPortion) else (rightPortion, leftPortion)

    if (omitFeatures.contains(featureIndex)) {
      hotPortion * hot.shapleyRecurse(input, omitFeatures, featureWeights) +
        coldPortion * cold.shapleyRecurse(input, omitFeatures, featureWeights)
    } else {
      // Check if the feature in this node has been used already (on the way down)
      val (hotFactor, coldFactor) = featureWeights.get(featureIndex) match {
        case None =>
          // If this is the first time the feature is being split on,
          // then when the feature is excluded the weights are hotPortion and coldPortion
          // and when the feature is included they are 1.0 and 0.0
          (
            FeatureWeightFactor(hotPortion, 1.0),
            FeatureWeightFactor(coldPortion, 0.0)
          )
        case Some(previousWeights) =>
          // If the feature has been split on before, then multiply the previous weights by the new ones
          (
            FeatureWeightFactor(hotPortion * previousWeights.weightWhenExcluded, previousWeights.weightWhenIncluded),
            FeatureWeightFactor(coldPortion * previousWeights.weightWhenExcluded, 0.0)
          )
      }

      // Add the features to the featureWeights map and recurse down
      val hotContrib = hot.shapleyRecurse(input, omitFeatures, featureWeights.updated(featureIndex, hotFactor))
      val coldContrib = cold.shapleyRecurse(input, omitFeatures, featureWeights.updated(featureIndex, coldFactor))

      // Simple sum on the way up
      coldContrib + hotContrib
    }
  }
}

case class ModelLeaf[+T](model: Model[T], depth: Int, trainingWeight: Double) extends ModelNode[T] {

  override def transform(input: Vector[AnyVal]): (PredictionResult[T], TreeMeta) =
    (model.transform(Seq(input)), TreeMeta(depth))

  /**
    * Compute the contribution to SHAP in the leaf based on the features that were encountered between the root node
    * and this leaf.  Note that the order of these features does *not* matter.
    *
    * The contributions are based on a tricky combinatorial factor that can be computed using dynamic programming.
    * For details of this procedure, see [[FeaturePowerSetTerms]].
    */
  def shapleyRecurse(
      input: Vector[AnyVal],
      omitFeatures: Set[Int],
      featureWeights: Map[Int, FeatureWeightFactor]
  ): DenseMatrix[Double] = {
    // Start with an empty matrix, into which we'll set the non-zero contributions
    val shapValues = DenseMatrix.zeros[Double](1, input.length)

    // Load all of the features into the set, which performs a dynamic programming calculation
    val set = new FeaturePowerSetTerms(featureWeights.size)
    featureWeights.values.foreach { case FeatureWeightFactor(exclude, include) => set.extend(exclude, include) }

    // The contribution is proportional to the leaf's prediction, so grab that
    this.model.transform(Seq(input)).expected.head match {
      case v: Double =>
        // For each feature, compute the contribution and store it in shapValues
        featureWeights.foreach {
          case (featureIndex, node) =>
            // Compute the weight of the contribution by removing this feature from the set and then computing the weights
            // These two steps are fused to avoid an extra memory allocation
            // This is equivalent to `set.unwind(node.weightWhenExcluded, node.weightWhenIncluded).totalWeight`
            val w = set.unwoundTotalWeight(node.weightWhenExcluded, node.weightWhenIncluded)
            shapValues(0, featureIndex) = w * (node.weightWhenIncluded - node.weightWhenExcluded) * v
        }
      case v: Char =>
        // For classification output is a class label. Calculate similar to regression.
        featureWeights.foreach {
          case (featureIndex, node) =>
            // Compute the weight of the contribution by removing this feature from the set and then computing the weights
            // These two steps are fused to avoid an extra memory allocation
            // This is equivalent to `set.unwind(node.weightWhenExcluded, node.weightWhenIncluded).totalWeight`
            val w = set.unwoundTotalWeight(node.weightWhenExcluded, node.weightWhenIncluded)
            shapValues(0, featureIndex) = w * (node.weightWhenIncluded - node.weightWhenExcluded) * v
        }

      case _ => throw new NotImplementedError()
    }
    shapValues
  }

  override def shapley(input: Vector[AnyVal], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = None
}
