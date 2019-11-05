package io.citrine.lolo.trees.splits

import io.citrine.lolo.trees.impurity.VarianceCalculator

import scala.collection.immutable.BitSet
import scala.collection.mutable
import scala.util.Random

/**
  * Find a split for a regression problem
  *
  * The splits are picked with a probability that is related to the reduction in variance:
  * P(split) ~ exp[ - {reduction in variance} / ({temperature} * {total variance}) ]
  * recalling that the "variance" here is weighted by the sample size (so its really the sum of the square difference
  * from the mean).  This is akin to kinetic monte carlo and simulated annealing techniques.
  *
  * Created by maxhutch on 11/29/16.
  */
case class BoltzmannSplitter(temperature: Double) extends Splitter[Double] {
  val minimumSupportedTemperature: Double = -1.0 / Math.log(Double.MinPositiveValue)
  require(temperature > minimumSupportedTemperature, s"Temperature must be > $minimumSupportedTemperature to avoid underflow")

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
    val beta = 1.0 / (temperature * initialVariance)

    val rep = data.head

    /* Try every feature index */
    val featureIndices: Seq[Int] = rep._1.indices

    val possibleSplits: Seq[(Split, Double, Double)] = Random.shuffle(featureIndices).take(numFeatures).map { index =>
      /* Use different spliters for each type */
      rep._1(index) match {
        case _: Double => BoltzmannSplitter.getBestRealSplit(data, calculator, index, minInstances, beta)
        case _: Char => BoltzmannSplitter.getBestCategoricalSplit(data, calculator, index, minInstances, beta)
        case _: Any => throw new IllegalArgumentException("Trying to split unknown feature type")
      }
    }

    val totalProbability = possibleSplits.map(_._3).sum
    if (totalProbability == 0) {
      (new NoSplit(), 0.0)
    } else {
      val draw = Random.nextDouble() * totalProbability
      var cumSum: Double = 0.0
      possibleSplits.foreach { case (split, variance, probability) =>
        cumSum = cumSum + probability
        if (draw < cumSum) {
          val deltaImpurity = initialVariance - variance
          return (split, deltaImpurity)
        }
      }
      // This shouldn't ever be hit
      throw new RuntimeException(s"Draw was beyond all the probabilities ${draw} ${totalProbability}")
    }
  }
}

object BoltzmannSplitter {
  /**
    * Find the best split on a continuous variable
    *
    * @param data        to split
    * @param index       of the feature to split on
    * @return the best split of this feature
    */
  def getBestRealSplit(
                        data: Seq[(Vector[AnyVal], Double, Double)],
                        calculator: VarianceCalculator,
                        index: Int,
                        minCount: Int,
                        beta: Double
                      ): (RealSplit, Double, Double) = {
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
        val score = Math.exp(-totalVariance * beta)
        val pivot = (left - right) * Random.nextDouble() + right
        Some(score, pivot, totalVariance)
      } else {
        None
      }
    }

    if (possibleSplits.isEmpty) {
      return (new RealSplit(index, 0.0), calculator.getImpurity, 0.0)
    }

    val totalScore = possibleSplits.map(_._1).sum
    if (totalScore > 0) {
      val draw = Random.nextDouble() * totalScore
      var cumSum: Double = 0.0
      possibleSplits.foreach { case (score, pivot, variance) =>
        cumSum = cumSum + score
        if (draw < cumSum) {
          return (new RealSplit(index, pivot), variance, totalScore)
        }
      }
      throw new RuntimeException(s"Draw was beyond all the probabilities: ${draw} > $cumSum")
    } else {
      val selected = possibleSplits(Random.nextInt(possibleSplits.size))
      (new RealSplit(index, selected._2), selected._3, totalScore)
    }
  }

  /**
    * Get find the best categorical splitter.
    *
    * @param data        to split
    * @param index       of the feature to split on
    * @return the best split of this feature
    */
  def getBestCategoricalSplit(
                               data: Seq[(Vector[AnyVal], Double, Double)],
                               calculator: VarianceCalculator,
                               index: Int,
                               minCount: Int,
                               beta: Double
                             ): (CategoricalSplit, Double, Double) = {
    /* Extract the features at the index */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val totalWeight = thinData.map(_._3).sum

    /* Group the data by categorical feature and compute the weighted sum and sum of the weights for each */
    val groupedData = thinData.groupBy(_._1).mapValues(g => (g.map(v => v._2 * v._3).sum, g.map(_._3).sum, g.size))

    /* Make sure there is more than one member for most of the classes */
    val nonTrivial: Double = groupedData.filter(_._2._3 > 1).map(_._2._2).sum
    if (nonTrivial / totalWeight < 0.5) {
      return (new CategoricalSplit(index, BitSet()), 0.0, 0.0)
    }

    /* Compute the average label for each categorical value */
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(p => p._1 / p._2)

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
        val score = Math.exp(- totalVariance * beta)
        val includeSet: mutable.BitSet = new mutable.BitSet() ++ orderedNames.slice(0, j + 1).map(_.toInt)
        Some((score, includeSet, totalVariance))
      } else {
        None
      }
    }

    if (possibleSplits.isEmpty) {
      return (new CategoricalSplit(index, new mutable.BitSet()), 0.0, 0.0)
    }

    val totalScore = possibleSplits.map{_._1}.sum
    // If none of the splits were valid, return a dummy split with no weight
    if (totalScore == 0.0) {
      return (new CategoricalSplit(index, new mutable.BitSet()), 0.0, 0.0)
    }

    val draw = Random.nextDouble() * totalScore
    var cumSum: Double = 0.0
    possibleSplits.foreach{case (score, includeSet, variance) =>
        cumSum = cumSum + score
        if (draw < cumSum) {
          return (new CategoricalSplit(index, includeSet), variance, totalScore)
        }
    }
    // This should never be hit
    throw new RuntimeException(s"Draw was beyond all the probabilities: $draw > $cumSum")
  }

}
