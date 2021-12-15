package io.citrine.lolo.trees.splits

import io.citrine.lolo.trees.impurity.MultiImpurityCalculator

import scala.util.Random

/**
  *
  * Created by maxhutch on 11/29/16.
  */
case class MultiTaskSplitter(randomizePivotLocation: Boolean = false, rng: Random = Random) extends Splitter[Array[AnyVal]] {

  /**
    * Get the best split, considering numFeature random features (w/o replacement)
    *
    * @param data        to split
    * @param numFeatures to consider, randomly
    * @param minInstances the minimum number of data points on a split node
    * @return a split object that optimally divides data
    */
  def getBestSplit(
                    data: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
                    numFeatures: Int,
                    minInstances: Int
                  ): (Split, Double) = {
    var bestSplit: Split = new NoSplit()
    var bestImpurity = Double.MaxValue
    val calculator = MultiImpurityCalculator.build(data.map(_._2), data.map(_._3))
    val initialImpurity = calculator.getImpurity

    /* Pre-compute these for the variance calculation */
    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices
    rng.shuffle(featureIndices).take(numFeatures).foreach { index =>

      /* Use different spliters for each type */
      val (possibleSplit, possibleImpurity) = rep._1(index) match {
        case _: Double => Splitter.getBestRealSplit[Array[AnyVal]](data, calculator, index, minInstances, randomizePivotLocation, rng)
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
    * Get find the best categorical splitter.
    *
    * @param data        to split
    * @param totalWeight Pre-computed data.map(d => d._3).sum
    * @param index       of the feature to split on
    * @return the best split of this feature
    */
  def getBestCategoricalSplit(
                               data: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
                               calculator: MultiImpurityCalculator,
                               index: Int,
                               minCount: Int
                             ): (Split, Double) = {
    /* Extract the features at the index */
    val thinData: Seq[(Char, Array[AnyVal], Double)] = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val totalWeight = thinData.map(_._3).sum

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData: Map[Char, (Double, Double, Double)] = thinData.groupBy(_._1).mapValues(g =>
      (computeImpurity(g.map(x => (x._2, x._3))), g.map(_._3).sum, g.size.toDouble)
    ).toMap

    /* Make sure there is more than one member for most of the classes */
    val nonTrivial: Double = groupedData.filter(_._2._3 > 1).map(_._2._2).sum
    if (nonTrivial / totalWeight < 0.25) {
      return (new NoSplit, Double.MaxValue)
    }

    /* Create an orderd list of the categories by average label */
    val orderedNames: Seq[Char] = groupedData.toSeq.sortBy(_._2._1).map(_._1)

    /* Add the categories one at a time in order of their average label */
    var leftNum = 0
    calculator.reset()
    val pivots = (0 until orderedNames.size).flatMap { j =>
      thinData.filter(r => orderedNames(j) == r._1).map { r =>
        calculator.add(r._2, r._3)
        leftNum = leftNum + 1
      }
      val totalImpurity = calculator.getImpurity

      if (leftNum >= minCount && thinData.size - leftNum >= minCount) {
        val set = orderedNames.take(j + 1).toSet
        Some(set, totalImpurity)
      } else {
        None
      }
    }
    if (pivots.isEmpty) {
      (new NoSplit, Double.MaxValue)
    } else {
      val best = pivots.minBy(_._2)
      (new CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ best._1.map(_.toInt)), best._2)
    }
  }

  /**
    * Compute the impurity of a set of weighted labels
    *
    * @param labels is a seq of (Array of multiple labels, single weight)
    * @return the impurity, which is in [0, number of labels * sum of weights]
    */
  def computeImpurity(labels: Seq[(Array[AnyVal], Double)]): Double = {
    val calculator = MultiImpurityCalculator.build(labels.map(_._1), labels.map(_._2))
    calculator.getImpurity
  }
}
