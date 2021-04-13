package io.citrine.lolo.trees.splits

import io.citrine.lolo.trees.impurity.VarianceCalculator
import io.citrine.lolo.trees.splits.BoltzmannSplitter.SplitterResult

import scala.collection.mutable
import scala.util.Random

/**
  * Find a split for a regression problem
  *
  * The splits are picked with a probability that is related to the reduction in variance:
  * P(split) ~ exp[ - {remaining variance} / ({temperature} * {total variance}) ]
  * recalling that the "variance" here is weighted by the sample size (so its really the sum of the square difference
  * from the mean of that side of the split).  This is analogous to simulated annealing and Metropolis-Hastings.
  *
  * The motivation here is to reduce the correlation of the trees by making random choices between splits that are
  * almost just as good as the strictly optimal one.  Reducing the correlation between trees will reduce the variance
  * in an ensemble method (e.g. random forests): the variance will both decrease more quickly with the tree count and
  * will reach a lower floor.  In this paragraph, we're using "variance" as in "bias-variance trade-off".
  *
  * Division by the local total variance make the splitting behavior invariant to data size and the scale of the labels.
  * That means, however, that you can't set the temperature based on a known absolute noise scale.  For that, you'd want
  * to divide by the total weight rather than the total variance.
  *
  * TODO: allow the rescaling to happen based on the total weight instead of the total variance, as an option
  *
  * Created by maxhutch on 11/29/16.
  *
  * @param temperature used to control how sensitive the probability of a split is to its change in variance.
  *                    The temperature can be thought of as a hyperparameter.
  */
case class BoltzmannSplitter(temperature: Double, rng: Random = Random) extends Splitter[Double] {
  require(temperature >= Float.MinPositiveValue, s"Temperature must be >= ${Float.MinPositiveValue} to avoid numerical underflows")

  /**
    * Get the a split probabalisticly, considering numFeature random features (w/o replacement), ensuring that the
    * resulting partitions have at least minInstances in them
    *
    * @param data         to split
    * @param numFeatures  to consider, randomly
    * @param minInstances minimum instances permitted in a post-split partition
    * @return a split object that optimally divides data
    */
  def getBestSplit(data: Seq[(Vector[AnyVal], Double, Double)], numFeatures: Int, minInstances: Int): (Split, Double) = {
    /* Pre-compute these for the variance calculation */
    val calculator = VarianceCalculator.build(data.map(_._2), data.map(_._3))
    val initialVariance = calculator.getImpurity

    // Don't split if there is no impurity to reduce
    if (initialVariance == 0) {
      return (new NoSplit(), 0.0)
    }
    val beta = 1.0 / (temperature * initialVariance)

    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices

    val possibleSplits: Seq[SplitterResult] = rng.shuffle(featureIndices).take(numFeatures).flatMap { index =>
      /* Use different spliters for each type */
      rep._1(index) match {
        case _: Double => BoltzmannSplitter.getBestRealSplit(data, calculator, index, minInstances, beta, rng)
        case _: Char => BoltzmannSplitter.getBestCategoricalSplit(data, calculator, index, minInstances, beta, rng)
        case _: Any => throw new IllegalArgumentException("Trying to split unknown feature type")
      }
    }

    // If we couldn't find a split, then return NoSplit with no variance reduction
    if (possibleSplits.isEmpty) {
      return (new NoSplit(), 0.0)
    }

    // Re-based the probabilities, such that the largest probability is order-1.0
    // This is meant to avoid every probability underflowing
    val rebase = possibleSplits.map(_.base).max
    val totalProbability = possibleSplits.map { x => x.rebasedScore * Math.exp(x.base - rebase) }.sum

    // select from a discrete probability distribution by drawing a random number and then computing the CDF
    // where the "draw" is the bin for which the CDF crosses the drawn number
    val draw = rng.nextDouble() * totalProbability
    // could be a scanLeft + find, but this is more readable
    var cumSum: Double = 0.0
    possibleSplits.foreach { case SplitterResult(split, variance, score, base) =>
      // Here's the probability rebasing again
      cumSum = cumSum + score * Math.exp(base - rebase)
      if (draw < cumSum) {
        val deltaImpurity = initialVariance - variance
        return (split, deltaImpurity)
      }
    }
    // This shouldn't ever be hit
    throw new RuntimeException(s"Draw was beyond all the probabilities ${draw} ${totalProbability}")
  }
}

object BoltzmannSplitter {

  /**
    * Container for function returns, like a decorated tuple
    *
    * The true score (proportional to draw probability) is rebasedScore * Math.exp(base).  This decomposition
    * is such that rebasedScore should always be >= 1.0.
    */
  protected case class SplitterResult(split: Split, variance: Double, rebasedScore: Double, base: Double) {
    // The rebasing procedure should result in rebasedScores that are >= 1.0 with finite bases
    // Otherwise, None should have been returned
    require(rebasedScore >= 1.0)
    require(!base.isNegInfinity)
  }

  /**
    * Find the best split on a continuous variable
    *
    * @param data  to split
    * @param calculator that will efficiently compute the impurity (variance in this case)
    * @param index of the feature to split on
    * @param minCount minimum number of training instances to leave in each of the children nodes
    * @param beta the inverse temperature (1.0 / (temperature * initial variance)) to scale the variances by
    * @return the best split of this feature, along with its score, base, and result variance
    */
  def getBestRealSplit(
                        data: Seq[(Vector[AnyVal], Double, Double)],
                        calculator: VarianceCalculator,
                        index: Int,
                        minCount: Int,
                        beta: Double,
                        rng: Random
                      ): Option[SplitterResult] = {
    /* Pull out the feature that's considered here and sort by it */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    val possibleSplits: Seq[(Double, Double, Double)] = (0 until data.size - minCount).flatMap { j =>
      val totalVariance = calculator.add(thinData(j)._2, thinData(j)._3)

      /* Keep track of the best split, avoiding splits in the middle of constant sets of feature values
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      val left = thinData(j + 1)._1
      val right = thinData(j)._1
      if (j + 1 >= minCount && Splitter.isDifferent(left, right)) {
        val score = -totalVariance * beta
        val pivot = (left - right) * rng.nextDouble() + right
        Some(score, pivot, totalVariance)
      } else {
        None
      }
    }

    if (possibleSplits.isEmpty) {
      return None
    }

    val base: Double = possibleSplits.map(_._1).max
    val totalScore = possibleSplits.map { case (s, _, _) => Math.exp(s - base) }.sum
    val draw = rng.nextDouble() * totalScore
    var cumSum: Double = 0.0
    possibleSplits.foreach { case (score, pivot, variance) =>
      cumSum = cumSum + Math.exp(score - base)
      if (draw < cumSum) {
        return Some(SplitterResult(new RealSplit(index, pivot), variance, totalScore, base))
      }
    }
    // This should never be hit; it would mean there's a bug in the logic above ^^
    throw new RuntimeException(s"Draw was beyond all the probabilities: ${draw} > $cumSum")
  }

  /**
    * Find the best categorical splitter.
    *
    * @param data  to split
    * @param calculator that will efficiently compute the impurity (variance in this case)
    * @param index of the feature to split on
    * @param minCount minimum number of training instances to leave in each of the children nodes
    * @param beta the inverse temperature (1.0 / (temperature * initial variance)) to scale the variances by
    * @return the best split of this feature, along with its score, base, and result variance
    */
  def getBestCategoricalSplit(
                               data: Seq[(Vector[AnyVal], Double, Double)],
                               calculator: VarianceCalculator,
                               index: Int,
                               minCount: Int,
                               beta: Double,
                               rng: Random
                             ): Option[SplitterResult] = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val totalWeight = thinData.map(_._3).sum

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData = thinData.groupBy(_._1).mapValues(g => (g.map(v => v._2 * v._3).sum, g.map(_._3).sum, g.size)).toMap

    /* Make sure there is more than one member for most of the classes */
    val nonTrivial: Double = groupedData.filter(_._2._3 > 1).map(_._2._2).sum
    if (nonTrivial / totalWeight < 0.5) {
      return None
    }

    /* Compute the average label for each categorical value */
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(p => p._1 / p._2).toMap

    /* Create an orderd list of the categories by average label */
    val orderedNames = categoryAverages.toSeq.sortBy(_._2).map(_._1)

    /* Base cases for the iteration */
    var leftNum: Int = 0

    /* Add the categories one at a time in order of their average label */
    calculator.reset()
    val possibleSplits: Seq[(Double, mutable.BitSet, Double)] = (0 until orderedNames.size - 1).flatMap { j =>
      val dat = groupedData(orderedNames(j))
      val totalVariance = calculator.add(dat._1 / dat._2, dat._2)
      leftNum += dat._3

      if (leftNum >= minCount && (thinData.size - leftNum) >= minCount) {
        val score = -totalVariance * beta
        val includeSet: mutable.BitSet = new mutable.BitSet() ++ orderedNames.slice(0, j + 1).map(_.toInt)
        Some((score, includeSet, totalVariance))
      } else {
        None
      }
    }

    if (possibleSplits.isEmpty) {
      return None
    }

    val base: Double = possibleSplits.map(_._1).max
    val totalScore = possibleSplits.map { case (s, _, _) => Math.exp(s - base) }.sum
    val draw = rng.nextDouble() * totalScore
    var cumSum: Double = 0.0
    possibleSplits.foreach { case (score, includeSet, variance) =>
      cumSum = cumSum + Math.exp(score - base)
      if (draw < cumSum) {
        return Some(SplitterResult(new CategoricalSplit(index, includeSet), variance, totalScore, base))
      }
    }
    // This should never be hit; it would mean there's a bug in the logic above ^^
    throw new RuntimeException(s"Draw was beyond all the probabilities: $draw > $cumSum")
  }
}
