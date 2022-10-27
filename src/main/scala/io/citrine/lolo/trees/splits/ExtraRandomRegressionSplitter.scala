package io.citrine.lolo.trees.splits

import io.citrine.lolo.api.TrainingRow
import io.citrine.random.Random
import io.citrine.lolo.trees.impurity.{ImpurityCalculator, VarianceCalculator}

/**
  * A splitter that defines Extremely Randomized Trees
  *
  * This is based on Geurts, P., Ernst, D. & Wehenkel, L. Extremely randomized trees. Mach Learn 63, 3–42 (2006).
  * https://doi.org/10.1007/s10994-006-6226-1.
  */
case class ExtraRandomRegressionSplitter() extends Splitter[Double] {

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
    rng.shuffle(featureIndices.toVector).take(numFeatures).foreach { index =>
      /* Use different spliters for each type */
      val (possibleSplit, possibleVariance) = rep.inputs(index) match {
        case _: Double => getBestRealSplit(data, calculator.copy(), index, minInstances, rng)
        case _: Char   => getBestCategoricalSplit(data, calculator.copy(), index, minInstances, rng)
        case _: Any    => throw new IllegalArgumentException("Trying to split unknown feature type")
      }

      /* Keep track of the best split */
      if (possibleVariance < bestVariance) {
        bestVariance = possibleVariance
        bestSplit = possibleSplit
      }
    }
    if (bestVariance >= initialVariance) {
      (NoSplit(), 0.0)
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
    * @param rng      random number generator, for reproducibility
    * @return the best split of this feature
    */
  def getBestRealSplit(
      data: Seq[TrainingRow[Double]],
      calculator: ImpurityCalculator[Double],
      index: Int,
      minCount: Int,
      rng: Random
  ): (RealSplit, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val axis: Seq[Double] = data.map(_.inputs(index).asInstanceOf[Double])
    val lowerBound = axis.min
    val upperBound = axis.max
    val pivot = lowerBound + (upperBound - lowerBound) * rng.nextDouble()

    val split = RealSplit(index, pivot)

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    data.foreach { dat =>
      if (split.turnLeft(dat.inputs)) {
        calculator.add(dat.label, dat.weight)
      }
    }

    (split, calculator.getImpurity)
  }

  /**
    * Get find the best categorical splitter.
    *
    * @param data  to split
    * @param index of the feature to split on
    * @param minCount minimum number of data points to allow in each of the resulting splits
    * @param rng      random number generator, for reproducibility
    * @return the best split of this feature
    */
  def getBestCategoricalSplit(
      data: Seq[TrainingRow[Double]],
      calculator: ImpurityCalculator[Double],
      index: Int,
      minCount: Int,
      rng: Random
  ): (Split, Double) = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat.inputs(index).asInstanceOf[Char], dat.label, dat.weight))

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData = thinData.groupBy(_._1).view.mapValues(g => (g.map(v => v._2 * v._3).sum, g.map(_._3).sum, g.size))

    /* Compute the average label for each categorical value */
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(p => p._1 / p._2).toMap

    /* Create an ordered list of the categories by average label */
    val orderedNames: Seq[Char] = categoryAverages.toSeq.sortBy(_._2).map(_._1)
    if (orderedNames.size == 1) {
      calculator.reset()
      return (NoSplit(), calculator.getImpurity)
    }

    val pivot = rng.nextInt(orderedNames.size - 1)
    val bestSet = orderedNames.slice(0, pivot + 1).toSet
    val split = CategoricalSplit(index, new scala.collection.mutable.BitSet() ++ bestSet.map(_.toInt))

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    data.foreach { dat =>
      if (split.turnLeft(dat.inputs)) {
        calculator.add(dat.label, dat.weight)
      }
    }

    (split, calculator.getImpurity)
  }

}
