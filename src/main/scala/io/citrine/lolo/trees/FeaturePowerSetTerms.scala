package io.citrine.lolo.trees

/** Description of a feature's effect on the weight ascribed to the node
  *
  * Decision trees can be thought of as weighted sums over their leaves. When there is knowledge of every feature, all
  * of the weight goes into a single leaf. However, when feature are excluded from the prediction, then both paths of
  * decisions that depend on those excluded features are assigned non-zero weights that are proportional to the share of
  * the training data that followed that path.
  *
  * This class captures those weights for a single feature and a single node. The `weightWhenIncluded` gives the factor
  * of the weight of the node due to this feature when the feature is included (i.e. known). It is always 0 or 1. The
  * `weightWhenExcluded` gives the factor of the weight of the node due to this feature when the feature is excluded
  * (i.e. unknown). It is always strictly greater than 0 and strictly less than 1.
  *
  * @param weightWhenExcluded
  *   factor of the weight of the leaf when the feature is excluded
  * @param weightWhenIncluded
  *   factor of the weight of the leaf when the feature is included
  */
case class FeatureWeightFactor(
    weightWhenExcluded: Double,
    weightWhenIncluded: Double
) {
  require(
    weightWhenIncluded == 0.0 || weightWhenIncluded == 1.0,
    s"Got weightWhenIncluded=$weightWhenIncluded, but should only ever be 0.0 or 1.0"
  )
  require(
    weightWhenExcluded > 0 && weightWhenExcluded < 1,
    s"Got weightWhenExcluded=$weightWhenExcluded, but should be > 0 and < 1"
  )
}

/** Weighted combinatorial factors corresponding to the power-set of features that are included
  *
  * The set is extended as new features are encountered. Each element in `weightBySubsetSize` corresponds to all of the
  * subsets of the features that are included vs excluded of a given size:
  *   - the 0th element corresponds to the term where all of the encountered features are excluded,
  *   - the 1st element sums the terms that have exactly one of the encountered feature included,
  *   - the 2nd element sums the terms that have exactly two of the encountered features included, and so on and so
  *     forth. The final element represents the single term where all of the features are "turned on".
  *
  * The calculation is performed efficiently using a dynamic programming technique that runs in quadratic time with the
  * number of features. It is described in the `extend` and `unwind` methods.
  *
  * @param maxFeatures
  *   number of features to allocate space for
  */
class FeaturePowerSetTerms(maxFeatures: Int) {
  // pre-allocation of this whole array is an attempted performance optimization.
  val weightBySubsetSize: Array[Double] = Array.fill[Double](maxFeatures + 1)(0.0)
  weightBySubsetSize(0) = 1.0
  var size: Int = 0

  /** Extend the power set by adding a new feature (in-place)
    *
    * Each subset weight is updated with a contribution from the subset of the same size multiplied by the weight when
    * the feature is excluded and the subset of size one smaller multiplied by the weight when the feature is included.
    * The additional terms in the expression track the combinatorial factors |S|! and (M - |S| - 1)! from eq. 2 of
    * https://arxiv.org/pdf/1802.03888.pdf
    *
    * @param weightWhenExcluded
    *   weight factor for this node when the feature is excluded (unknown)
    * @param weightWhenIncluded
    *   weight factor for this node when the feature is included (known)
    * @return
    *   this (in-place)
    */
  def extend(
      weightWhenExcluded: Double,
      weightWhenIncluded: Double
  ): FeaturePowerSetTerms = {
    size += 1

    // use a single scratch variable so we can do a forward in-place update
    var register: Double = weightBySubsetSize(0)

    // there is no subset with size -1, so special case i = 0
    weightBySubsetSize(0) = weightWhenExcluded * weightBySubsetSize(0) * ((size - 0).toDouble / (size + 1))

    (1 until size + 1).foreach { i =>
      val contrib = weightWhenIncluded * register * (i.toDouble / (size + 1))
      // stash the weight for use in the next iteration
      register = weightBySubsetSize(i)
      weightBySubsetSize(i) = weightWhenExcluded * weightBySubsetSize(i) * ((size - i).toDouble / (size + 1)) + contrib
    }

    this
  }

  /** Remove a feature from the power set, reversing the action of the `extend` method
    *
    * This method is probably better called "remove", but it is called unwind in the paper. It's not pretty, but it
    * simply inverts `extend`. If you think about extend as multiplying by a matrix with weightWhenExcluded... on the
    * diagonal and weightWhenIncluded... on the lower diagonal. This method is performing a fast linear solve on that
    * banded system, with the two logical branches corresponding to the banded vs purely diagonal case. The ... above
    * denotes some combinatorial terms also present.
    *
    * @return
    *   feature power set with a feature removed (out-of-place)
    */
  def unwind(weightWhenExcluded: Double, weightWhenIncluded: Double): FeaturePowerSetTerms = {
    // make a copy so this is out of place
    val out = this.copy()

    // reverse the procedure in extend
    var n = out.weightBySubsetSize(size)
    (size - 1 to 0 by -1).foreach { j =>
      if (weightWhenIncluded != 0.0) {
        val t = out.weightBySubsetSize(j)
        out.weightBySubsetSize(j) = n * (size + 1) / ((j + 1) * weightWhenIncluded)
        n = t - out.weightBySubsetSize(j) * weightWhenExcluded * ((size - j).toDouble / (size + 1))
      } else {
        out.weightBySubsetSize(j) = out.weightBySubsetSize(j) * (size + 1).toDouble / (weightWhenExcluded * (size - j))
      }
    }

    // bookkeeping
    out.size -= 1
    out
  }

  /** Get the total weight of the feature power set, as a sum over the weights by size of the sets in the power set
    */
  def totalWeight: Double = weightBySubsetSize.take(size + 1).sum

  /** This is an optimized version of unwind().totalWeight. Because the unwound feature power set isn't used, it is not
    * even allocated. That avoided allocation is a significant performance improvement.
    */
  def unwoundTotalWeight(weightWhenExcluded: Double, weightWhenIncluded: Double): Double = {
    var res = 0.0
    // reverse the procedure in extend

    if (weightWhenIncluded != 0.0) {
      var n = weightBySubsetSize(size)
      (size - 1 to 0 by -1).foreach { j =>
        val x = n * (size + 1) / ((j + 1) * weightWhenIncluded)
        res += x
        n = weightBySubsetSize(j) - x * weightWhenExcluded * (size - j) / (size + 1)
      }
    } else {
      (0 until size).foreach { j =>
        res += weightBySubsetSize(j) * (size + 1) / (weightWhenExcluded * (size - j))
      }
    }
    res
  }

  def copy(): FeaturePowerSetTerms = {
    val newSet = new FeaturePowerSetTerms(this.maxFeatures)
    newSet.size = this.size
    this.weightBySubsetSize.zipWithIndex.foreach { case (x, i) => newSet.weightBySubsetSize(i) = x }
    newSet
  }
}
