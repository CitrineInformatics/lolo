package io.citrine.lolo.trees.splits

import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.trees.impurity.VarianceCalculator
import io.citrine.random.Random

import scala.collection.immutable.BitSet

/**
  * Find the best split for regression problems.
  *
  * The best split is the one that reduces the total weighted variance:
  * totalVariance = N_left * \sigma_left^2 + N_right * \sigma_right^2
  * which, in scala-ish, would be:
  * totalVariance = leftWeight  * (leftSquareSum /leftWeight  - (leftSum  / leftWeight )^2)
  * + rightWeight * (rightSquareSum/rightWeight - (rightSum / rightWeight)^2)
  * Because we are comparing them, we can subtract off leftSquareSum + rightSquareSum, which yields the following simple
  * expression after some simplification:
  * totalVariance = -leftSum * leftSum / leftWeight - Math.pow(totalSum - leftSum, 2) / (totalWeight - leftWeight)
  * which depends only on updates to leftSum and leftWeight (since totalSum and totalWeight are constant).
  *
  * Created by maxhutch on 11/29/16.
  */
case class RegressionSplitter(randomizePivotLocation: Boolean = false) extends Splitter[Double] {

  /**
    * Get the best split, considering numFeature random features (w/o replacement)
    *
    * @param data        to split
    * @param numFeatures to consider, randomly
    * @param minInstances minimum instances permitted in a post-split partition
    * @param rng          random number generator for reproducibility
    * @return a split object that optimally divides data
    */
  def getBestSplit(
      data: Seq[TrainingRow[Double]],
      numFeatures: Int,
      minInstances: Int,
      rng: Random
  ): (Split, Double) = {

    val calculator = VarianceCalculator.build(data.map(_.label), data.map(_.weight))
    val initialVariance = calculator.getImpurity
    var bestSplit: Split = NoSplit()
    var bestVariance = Double.MaxValue

    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep.inputs.indices
    rng.shuffle(featureIndices).take(numFeatures).foreach { index =>
      /* Use different splitters for each type */
      val (possibleSplit, possibleVariance) = rep.inputs(index) match {
        case _: Double =>
          Splitter.getBestRealSplit[Double](data, calculator.copy(), index, minInstances, randomizePivotLocation, rng)
        case _: Char => getBestCategoricalSplit(data, calculator.copy(), index, minInstances)
        case _: Any  => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleVariance < bestVariance) {
        bestVariance = possibleVariance
        bestSplit = possibleSplit
      }
    }
    if (bestVariance == Double.MaxValue) {
      (NoSplit(), 0.0)
    } else {
      val deltaImpurity = initialVariance - bestVariance
      (bestSplit, deltaImpurity)
    }
  }

  /**
    * Get find the best categorical splitter.
    *
    * @param data  to split
    * @param index of the feature to split on
    * @return the best split of this feature
    */
  def getBestCategoricalSplit(
      data: Seq[TrainingRow[Double]],
      calculator: VarianceCalculator,
      index: Int,
      minCount: Int
  ): (CategoricalSplit, Double) = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat.inputs(index).asInstanceOf[Char], dat.label, dat.weight))
    val totalWeight = thinData.map(_._3).sum

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData = thinData.groupBy(_._1).view.mapValues(g => (g.map(v => v._2 * v._3).sum, g.map(_._3).sum, g.size))

    /* Make sure there is more than one member for most of the classes */
    val nonTrivial: Double = groupedData.filter(_._2._3 > 1).map(_._2._2).sum
    if (nonTrivial / totalWeight < 0.5) {
      return (CategoricalSplit(index, BitSet()), Double.MaxValue)
    }

    /* Compute the average label for each categorical value */
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(p => p._1 / p._2).toMap

    /* Create an ordered list of the categories by average label */
    val orderedNames = categoryAverages.toSeq.sortBy(_._2).map(_._1)

    /* Base cases for the iteration */
    var leftNum: Int = 0
    var bestVariance = Double.MaxValue
    var bestSet = Set.empty[Char]

    /* Add the categories one at a time in order of their average label */
    calculator.reset()
    (0 until orderedNames.size - 1).foreach { j =>
      val dat = groupedData(orderedNames(j))
      val totalVariance = calculator.add(dat._1 / dat._2, dat._2)

      leftNum = leftNum + dat._3

      /* Keep track of the best split */
      if (totalVariance < bestVariance && leftNum >= minCount && (thinData.size - leftNum) >= minCount) {
        bestVariance = totalVariance
        bestSet = orderedNames.slice(0, j + 1).toSet
      }
    }
    (CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ bestSet.map(_.toInt)), bestVariance)
  }

}
