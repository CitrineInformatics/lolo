package io.citrine.lolo.trees.splits

import scala.collection.mutable
import scala.util.Random

/**
  * Find the best split for classification problems.
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
  * Created by maxhutch on 12/2/16.
  */
object ClassificationSplitter {

  /**
    * Get the best split, considering numFeature random features (w/o replacement)
    *
    * @param data        to split
    * @param numFeatures to consider, randomly
    * @return a split object that optimally divides data
    */
  def getBestSplit(data: Seq[(Vector[AnyVal], Char, Double)], numFeatures: Int): (Split, Double) = {
    var bestSplit: Split = NoSplit
    var bestImpurity = Double.MaxValue

    /* Pre-compute these for the variance calculation */
    val totalCategoryWeights = data.map(d => (d._2, d._3)).groupBy(_._1).mapValues(_.map(_._2).sum)
    val totalSquareSum = totalCategoryWeights.map(v => Math.pow(v._2, 2)).sum
    val totalWeight = totalCategoryWeights.map(_._2).sum

    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices
    Random.shuffle(featureIndices).take(numFeatures).foreach { index =>

      /* Use different spliters for each type */
      val (possibleSplit, possibleImpurity) = rep._1(index) match {
        case x: Double => getBestRealSplit(data, totalCategoryWeights, totalWeight, totalSquareSum, index)
        case x: Char => getBestCategoricalSplit(data, totalCategoryWeights, totalWeight, totalSquareSum, index)
        case x: Any => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleImpurity < bestImpurity) {
        bestImpurity = possibleImpurity
        bestSplit = possibleSplit
      }
    }
    val deltaImpurity = - bestImpurity
    (bestSplit, deltaImpurity)
  }

  /**
    * Find the best split on a continuous variable
    *
    * @param data        to split
    * @param totalCategoryWeights    Pre-computed data.map(d => data._2 * data._3).sum
    * @param totalWeight Pre-computed data.map(d => d._3).sum
    * @param index       of the feature to split on
    * @return the best split of this feature
    */
  def getBestRealSplit(data: Seq[(Vector[AnyVal], Char, Double)], totalCategoryWeights: Map[Char, Double], totalWeight: Double, totalSquareSum: Double, index: Int): (RealSplit, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)

    /* Base cases for iteration */
    val leftCategoryWeights = mutable.Map[Char, Double]()
    var leftWeight = 0.0
    var leftSquareSum = 0.0
    var rightSquareSum = totalSquareSum

    var bestPurity = Double.MinValue
    var bestPivot = Double.MinValue

    /* Move the data from the right to the left partition one value at a time */
    (0 until data.size - 1).foreach { j =>
      val y = thinData(j)._2
      val w = thinData(j)._3
      val wl = leftCategoryWeights.getOrElse(y, 0.0)
      leftSquareSum +=  w * (w + 2 * wl)
      rightSquareSum += w * (w - 2 * (totalCategoryWeights(y) - wl))
      leftCategoryWeights(y) = w + wl
      leftWeight = leftWeight + w

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalPurity = leftSquareSum / leftWeight + rightSquareSum / (totalWeight - leftWeight)

      /* Keep track of the best split, avoiding splits in the middle of constant sets of feature values
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      if (totalPurity > bestPurity && thinData(j + 1)._1 > thinData(j)._1 + 1.0e-9) {
        bestPurity = totalPurity
        /* Try pivots at the midpoints between consecutive member values */
        bestPivot = (thinData(j + 1)._1 + thinData(j)._1) / 2.0 // thinData(j)._1 //
      }
    }
    (new RealSplit(index, bestPivot), totalWeight - bestPurity)
  }

  def getBestCategoricalSplit(data: Seq[(Vector[AnyVal], Char, Double)], totalCategoryWeights: Map[Char, Double], totalWeight: Double, totalSquareSum: Double, index: Int): (CategoricalSplit, Double) = {
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val groupedData = thinData.groupBy(_._1).mapValues{g =>
      val dict = g.groupBy(_._2).mapValues(v => v.map(_._3).sum)
      val impurity = dict.values.map(Math.pow(_, 2)).sum / Math.pow(dict.values.sum, 2)
      (dict, impurity)
    }
    val orderedNames = groupedData.toSeq.sortBy(_._2._2).map(_._1)

    /* Base cases for iteration */
    val leftCategoryWeights = mutable.Map[Char, Double]()
    var leftWeight = 0.0
    var leftSquareSum = 0.0
    var rightSquareSum = totalSquareSum

    var bestPurity = Double.MinValue
    var bestSet = Set.empty[Char]

    /* Move the data from the right to the left partition one value at a time */
    (0 until orderedNames.size - 1).foreach { j =>
      val dict = groupedData(orderedNames(j))._1
      dict.foreach { case (y, w) =>
        val wl = leftCategoryWeights.getOrElse(y, 0.0)
        leftSquareSum += w * (w + 2 * wl)
        rightSquareSum += w * (w - 2 * (totalCategoryWeights(y) - wl))
        leftCategoryWeights(y) = w + wl
        leftWeight = leftWeight + w
      }

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalPurity = leftSquareSum / leftWeight + rightSquareSum / (totalWeight - leftWeight)

      /* Keep track of the best split, avoiding splits in the middle of constant sets of feature values
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      if (totalPurity > bestPurity ) {
        bestPurity = totalPurity
        bestSet = orderedNames.slice(0, j + 1).toSet
      }
    }
    (new CategoricalSplit(index, bestSet), totalWeight - bestPurity)
  }

}
