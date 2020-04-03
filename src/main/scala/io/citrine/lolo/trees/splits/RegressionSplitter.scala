package io.citrine.lolo.trees.splits

import io.citrine.lolo.trees.impurity.VarianceCalculator

import scala.collection.immutable.BitSet
import scala.util.Random

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
case class RegressionSplitter(randomizePivotLocation: Boolean = false, rng: Random = Random) extends Splitter[Double] {

  /**
    * Get the best split, considering numFeature random features (w/o replacement)
    *
    * @param data        to split
    * @param numFeatures to consider, randomly
    * @return a split object that optimally divides data
    */
  def getBestSplit(
                    data: Seq[(Vector[AnyVal], Double, Double)],
                    numFeatures: Int,
                    minInstances: Int
                  ): (Split, Double) = {

    val calculator = VarianceCalculator.build(data.map(_._2), data.map(_._3))
    val initialVariance = calculator.getImpurity
    var bestSplit: Split = new NoSplit()
    var bestVariance = Double.MaxValue

    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices
    rng.shuffle(featureIndices).take(numFeatures).foreach { index =>

      /* Use different spliters for each type */
      val (possibleSplit, possibleVariance) = rep._1(index) match {
        case _: Double => getBestRealSplit(data, calculator.copy(), index, minInstances, randomizePivotLocation)
        case _: Char => getBestCategoricalSplit(data, calculator.copy(), index, minInstances)
        case _: Any => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleVariance < bestVariance) {
        bestVariance = possibleVariance
        bestSplit = possibleSplit
      }
    }
    if (bestVariance == Double.MaxValue) {
      (new NoSplit(), 0.0)
    } else {
      val deltaImpurity = initialVariance - bestVariance
      (bestSplit, deltaImpurity)
    }
  }

  /**
    * Find the best split on a continuous variable.
    *
    * If randomizePivotLocation is true, the split pivots are drawn from a uniform random distribution between the two
    * data points.  Each such pivot results in the same data split, but randomization can improve generalizability,
    * particularly as part of an ensemble (i.e. random forests).
    *
    * @param data  to split
    * @param index of the feature to split on
    * @param minCount minimum number of data points to allow in each of the resulting splits
    * @param randomizePivotLocation whether generate splits randomly between the data points (default: false)
    * @return the best split of this feature
    */
  def getBestRealSplit(
                        data: Seq[(Vector[AnyVal], Double, Double)],
                        calculator: VarianceCalculator,
                        index: Int,
                        minCount: Int,
                        randomizePivotLocation: Boolean = false
                      ): (RealSplit, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)


    /* Base cases for iteration */
    var bestVariance = Double.MaxValue
    var bestPivot = Double.MinValue

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    (0 until data.size - minCount).foreach { j =>
      val totalVariance = calculator.add(thinData(j)._2, thinData(j)._3)

      /* Keep track of the best split, avoiding splits in the middle of constant sets of feature values
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      val left = thinData(j + 1)._1
      val right = thinData(j)._1
      if (totalVariance < bestVariance && j + 1 >= minCount && Splitter.isDifferent(left, right)) {
        bestVariance = totalVariance
        /* Try pivots at the midpoints between consecutive member values */
        bestPivot = if (randomizePivotLocation) {
          (left - right) * rng.nextDouble() + right
        } else {
          (left + right) / 2.0
        }
      }
    }
    (new RealSplit(index, bestPivot), bestVariance)
  }

  /**
    * Get find the best categorical splitter.
    *
    * @param data  to split
    * @param index of the feature to split on
    * @return the best split of this feature
    */
  def getBestCategoricalSplit(
                               data: Seq[(Vector[AnyVal], Double, Double)],
                               calculator: VarianceCalculator,
                               index: Int,
                               minCount: Int
                             ): (CategoricalSplit, Double) = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val totalWeight = thinData.map(_._3).sum

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData = thinData.groupBy(_._1).mapValues(g => (g.map(v => v._2 * v._3).sum, g.map(_._3).sum, g.size))

    /* Make sure there is more than one member for most of the classes */
    val nonTrivial: Double = groupedData.filter(_._2._3 > 1).map(_._2._2).sum
    if (nonTrivial / totalWeight < 0.5) {
      return (new CategoricalSplit(index, BitSet()), Double.MaxValue)
    }

    /* Compute the average label for each categorical value */
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(p => p._1 / p._2)

    /* Create an orderd list of the categories by average label */
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
    (new CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ bestSet.map(_.toInt)), bestVariance)
  }

}
