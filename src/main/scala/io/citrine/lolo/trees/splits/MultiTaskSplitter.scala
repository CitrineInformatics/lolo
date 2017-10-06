package io.citrine.lolo.trees.splits

import scala.util.Random

/**
  *
  * Created by maxhutch on 11/29/16.
  */
object MultiTaskSplitter {

  /**
    * Get the best split, considering numFeature random features (w/o replacement)
    *
    * @param data        to split
    * @param numFeatures to consider, randomly
    * @return a split object that optimally divides data
    */
  def getBestSplit(data: Seq[(Vector[AnyVal], Array[AnyVal], Double)], numFeatures: Int, minInstances: Int): (Split, Double) = {
    var bestSplit: Split = new NoSplit()
    var bestImpurity = Double.MaxValue
    val totalWeight = data.map(_._3).sum

    /* Pre-compute these for the variance calculation */
    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices
    Random.shuffle(featureIndices).take(numFeatures).foreach { index =>

      /* Use different spliters for each type */
      val (possibleSplit, possibleImpurity) = rep._1(index) match {
        case _: Double => getBestRealSplit(data, totalWeight, index, minInstances)
        case _: Char => getBestCategoricalSplit(data, totalWeight, index, minInstances)
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
      val deltaImpurity = computeImpurity(data.map(x => (x._2, x._3))) - bestImpurity
      (bestSplit, deltaImpurity)
    }
  }

  /**
    * Find the best split on a continuous variable
    *
    * @param data  to split
    * @param index of the feature to split on
    * @return the best split of this feature
    */
  def getBestRealSplit(
                        data: Seq[(Vector[AnyVal], Array[AnyVal], Double)],
                        totalWeight: Double,
                        index: Int,
                        minCount: Int
                      ): (Split, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)
    val features = thinData.map(x => x._1)
    val labels = thinData.map(x => (x._2, x._3))

    /* Move the data from the right to the left partition one value at a time */
    val pivots = (minCount until data.size - (minCount - 1)).flatMap { j =>
      if (Math.abs((features(j) - features(j - 1)) / features(j - 1)) > 1.0e-9) {
        val totalImpurity = computeImpurity(labels.take(j)) + computeImpurity(labels.takeRight(data.size - j))
        val pivot = (features(j) + features(j - 1)) / 2.0
        Some((pivot, totalImpurity))
      } else {
        None
      }
    }
    if (pivots.isEmpty) {
      (new NoSplit, Double.MaxValue)
    } else {
      val best = pivots.minBy(_._2)
      (new RealSplit(index, best._1), best._2)
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
                               totalWeight: Double,
                               index: Int,
                               minCount: Int
                             ): (Split, Double) = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData: Map[Char, (Double, Double, Double)] = thinData.groupBy(_._1).mapValues(g =>
      (computeImpurity(g.map(x => (x._2, x._3))), g.map(_._3).sum, g.size)
    )

    /* Make sure there is more than one member for most of the classes */
    val nonTrivial: Double = groupedData.filter(_._2._3 > 1).map(_._2._2).sum
    if (nonTrivial / totalWeight < 0.25) {
      return (new NoSplit, Double.MaxValue)
    }

    /* Create an orderd list of the categories by average label */
    val orderedNames: Seq[Char] = groupedData.toSeq.sortBy(_._2._1).map(_._1)

    /* Add the categories one at a time in order of their average label */
    val pivots = (1 until orderedNames.size).flatMap { j =>
      val (left, right) = thinData.partition(x => orderedNames.take(j).contains(x._1))

      if (left.size >= minCount && right.size >= minCount) {
        val totalImpurity = Seq(left, right).map(part => computeImpurity(part.map(x => (x._2, x._3)))).sum
        val set = orderedNames.take(j).toSet
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
    * @param labels is a seq of (Array of multiple labels, single weight)
    * @return the impurity, which is in [0, number of labels * sum of weights]
    */
  def computeImpurity(labels: Seq[(Array[AnyVal], Double)]): Double = {
    // Return early if there is no impurity
    if (labels.size == 1 || labels.head._1.isEmpty || labels.map(_._2).sum == 0.0) return 0.0

    // Sum the impurity of each individual label
    labels.head._1.indices.map { i =>
      // Check type
      labels.head._1(i) match {
        // Compute the Sum of weight * (x - mean)^2
        case _: Double =>
          val filtered = labels.map(x => (x._1(i).asInstanceOf[Double], x._2)).filterNot(_._1.isNaN)
          val sumWeights = filtered.map(_._2).sum
          val mean: Double = filtered.map(l => l._1 * l._2).sum / sumWeights
          filtered.map(l => l._2 * Math.pow(l._1 - mean, 2.0)).sum
        // Compute the Gini impurity, then multiply it by the total weight
        case _: Char =>
          val filtered = labels.map(x => (x._1(i).asInstanceOf[Char], x._2)).filter(_._1 > 0)
          val sumWeights = filtered.map(_._2).sum
          (1.0 - filtered.groupBy(_._1).values
            .map(x => Math.pow(x.map(_._2).sum / sumWeights, 2.0)).sum) * sumWeights
      }
    }.sum
  }
}
