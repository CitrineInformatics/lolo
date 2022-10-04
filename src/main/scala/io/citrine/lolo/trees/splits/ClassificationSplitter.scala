package io.citrine.lolo.trees.splits

import io.citrine.random.Random
import io.citrine.lolo.trees.impurity.GiniCalculator

/**
  * Find the best split for classification problems.
  *
  * Created by maxhutch on 12/2/16.
  */
case class ClassificationSplitter(randomizedPivotLocation: Boolean = false) extends Splitter[Char] {

  /**
    * Get the best split, considering numFeature random features (w/o replacement)
    *
    * @param data         to split
    * @param numFeatures  to consider, randomly
    * @param minInstances minimum instances permitted in a post-split partition
    * @param rng          random number generator for reproducibility
    * @return a split object that optimally divides data
    */
  def getBestSplit(
      data: Seq[(Vector[AnyVal], Char, Double)],
      numFeatures: Int,
      minInstances: Int,
      rng: Random
  ): (Split, Double) = {
    var bestSplit: Split = NoSplit()
    var bestImpurity = Double.MaxValue

    /* Pre-compute these for the variance calculation */
    val calculator = GiniCalculator.build(data.map(p => (p._2, p._3)))
    val initialImpurity = calculator.getImpurity

    val rep = data.head
    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices
    rng.shuffle(featureIndices).take(numFeatures).foreach { index =>
      /* Use different spliters for each type */
      val (possibleSplit, possibleImpurity) = rep._1(index) match {
        case _: Double =>
          Splitter.getBestRealSplit[Char](data, calculator, index, minInstances, randomizedPivotLocation, rng)
        case _: Char => getBestCategoricalSplit(data, calculator, index, minInstances)
        case _: Any  => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleImpurity < bestImpurity) {
        bestImpurity = possibleImpurity
        bestSplit = possibleSplit
      }
    }
    if (bestImpurity == Double.MaxValue) {
      (NoSplit(), 0.0)
    } else {
      val deltaImpurity = initialImpurity - bestImpurity
      (bestSplit, deltaImpurity)
    }
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
      dict.foreach {
        case (y, w) =>
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
    (CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ bestSet.map(_.toInt)), bestPurity)
  }

}
