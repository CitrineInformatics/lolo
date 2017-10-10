package io.citrine.lolo.trees.splits

import scala.collection.mutable
import scala.util.Random

/**
  * Find the best split for classification problems.
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
  def getBestSplit(data: Seq[(Vector[AnyVal], Char, Double)], numFeatures: Int, minInstances: Int): (Split, Double) = {
    var bestSplit: Split = new NoSplit()
    var bestImpurity = Double.MaxValue

    /* Pre-compute these for the variance calculation */
    val calculator = GiniCalculator.build(data.map(p => (p._2, p._3)))
    val initialImpurity = calculator.getImpurity

    val rep = data.head
    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices
    Random.shuffle(featureIndices).take(numFeatures).foreach { index =>

      /* Use different spliters for each type */
      val (possibleSplit, possibleImpurity) = rep._1(index) match {
        case _: Double => getBestRealSplit(data, calculator, index, minInstances)
        case _: Char => getBestCategoricalSplit(data, calculator, index, minInstances)
        case _: Any => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleImpurity < bestImpurity) {
        bestImpurity = possibleImpurity
        bestSplit = possibleSplit
      }
    }
    if (bestImpurity == Double.MaxValue) {
      (new NoSplit(), 0.0)
    } else {
      val deltaImpurity = initialImpurity - bestImpurity
      (bestSplit, deltaImpurity)
    }
  }

  /**
    * Find the best split on a continuous variable
    *
    * @param data                 to split
    * @param totalCategoryWeights Pre-computed data.map(d => data._2 * data._3).sum
    * @param totalWeight          Pre-computed data.map(d => d._3).sum
    * @param index                of the feature to split on
    * @return the best split of this feature
    */
  def getBestRealSplit(
                        data: Seq[(Vector[AnyVal], Char, Double)],
                        calculator: GiniCalculator,
                        index: Int,
                        minCount: Int
                      ): (RealSplit, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)

    var bestPurity = Double.MaxValue
    var bestPivot = Double.MinValue

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    (0 until data.size - minCount).foreach { j =>
      val totalPurity = calculator.add(thinData(j)._2, thinData(j)._3)

      /* Keep track of the best split, avoiding splits in the middle of constant sets of feature values
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      if (totalPurity < bestPurity && j + 1 >= minCount && Math.abs((thinData(j + 1)._1 - thinData(j)._1)/thinData(j)._1) > 1.0e-9) {
        bestPurity = totalPurity
        /* Try pivots at the midpoints between consecutive member values */
        bestPivot = (thinData(j + 1)._1 + thinData(j)._1) / 2.0 // thinData(j)._1 //
      }
    }
    (new RealSplit(index, bestPivot), bestPurity)
  }

  def getBestCategoricalSplit(
                               data: Seq[(Vector[AnyVal], Char, Double)],
                               calculator: GiniCalculator,
                               index: Int,
                               minCount: Int
                             ): (CategoricalSplit, Double) = {
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val groupedData = thinData.groupBy(_._1).mapValues { g =>
      val dict = g.groupBy(_._2).mapValues(v => v.map(_._3).sum)
      val impurity = dict.values.map(Math.pow(_, 2)).sum / Math.pow(dict.values.sum, 2)
      (dict, impurity, g.size)
    }
    val orderedNames = groupedData.toSeq.sortBy(_._2._2).map(_._1)

    /* Base cases for iteration */
    var leftNum = 0
    var bestPurity = Double.MaxValue
    var bestSet = Set.empty[Char]

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    (0 until orderedNames.size - 1).foreach { j =>
      val gd = groupedData(orderedNames(j))
      val dict = gd._1
      dict.foreach { case (y, w) =>
        calculator.add(y, w)
      }
      leftNum = leftNum + gd._3

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalPurity = calculator.getImpurity

      /* Keep track of the best split, avoiding splits in the middle of constant sets of feature values
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      if (totalPurity < bestPurity && leftNum >= minCount && (thinData.size - leftNum) >= minCount) {
        bestPurity = totalPurity
        bestSet = orderedNames.slice(0, j + 1).toSet
      }
    }
    (new CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ bestSet.map(_.toInt)), bestPurity)
  }

}
