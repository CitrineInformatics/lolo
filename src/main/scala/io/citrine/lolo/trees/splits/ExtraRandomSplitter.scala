package io.citrine.lolo.trees.splits

import io.citrine.lolo.trees.impurity.VarianceCalculator

import scala.util.Random

/**
  * Find the best split for regression problems.
  *
  * @param rng random number generator
  */
case class ExtraRandomSplitter(
                               rng: Random = Random
                              ) extends Splitter[Double] {

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
    rng.shuffle(featureIndices.toVector).take(numFeatures).foreach { index =>
      /* Use different spliters for each type */
      val (possibleSplit, possibleVariance) = rep._1(index) match {
        case _: Double => getBestRealSplit(data, calculator.copy(), index, minInstances)
        case _: Char => getBestCategoricalSplit(data, calculator.copy(), index, minInstances)
        case _: Any => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleVariance < bestVariance) {
        bestVariance = possibleVariance
        bestSplit = possibleSplit
      }
    }
    if (bestVariance >=  initialVariance) {
      (new NoSplit(), 0.0)
    } else {
      val deltaImpurity = initialVariance - bestVariance
      (bestSplit, deltaImpurity)
    }
  }

  /**
    * Find the best split on a continuous variable.
    *
    * @param data  to split
    * @param index of the feature to split on
    * @param minCount minimum number of data points to allow in each of the resulting splits
    * @return the best split of this feature
    */
  def getBestRealSplit(
                        data: Seq[(Vector[AnyVal], Double, Double)],
                        calculator: VarianceCalculator,
                        index: Int,
                        minCount: Int
                      ): (RealSplit, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val axis: Seq[Double] = data.map(_._1(index).asInstanceOf[Double])
    val lowerBound = axis.min
    val upperBound = axis.max
    val pivot = lowerBound + (upperBound - lowerBound) * rng.nextDouble()

    val split = new RealSplit(index, pivot)

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    data.foreach{ dat =>
      if (split.turnLeft(dat._1)) {
        calculator.add(dat._2, dat._3)
      }
    }

    (split, calculator.getImpurity)
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
                             ): (Split, Double) = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData = thinData.groupBy(_._1).mapValues(g => (g.map(v => v._2 * v._3).sum, g.map(_._3).sum, g.size))

    /* Compute the average label for each categorical value */
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(p => p._1 / p._2)

    /* Create an ordered list of the categories by average label */
    val orderedNames: Seq[Char] = categoryAverages.toSeq.sortBy(_._2).map(_._1)
    if (orderedNames.size == 1) {
      calculator.reset()
      return (new NoSplit(), calculator.getImpurity)
    }

    val pivot = rng.nextInt(orderedNames.size - 1)
    val bestSet = orderedNames.slice(0, pivot + 1).toSet
    val split = new CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ bestSet.map(_.toInt))

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    data.foreach{ dat =>
      if (split.turnLeft(dat._1)) {
        calculator.add(dat._2, dat._3)
      }
    }

    (split, calculator.getImpurity)
  }

}
