package io.citrine.lolo.trees.splits

import io.citrine.lolo.trees.impurity.ImpurityCalculator

import scala.util.Random

/**
  * Created by maxhutch on 7/5/17.
  */
trait Splitter[T] {
  def getBestSplit(data: Seq[(Vector[AnyVal], T, Double)], numFeatures: Int, minInstances: Int): (Split, Double)
}

object Splitter {
  def isDifferent(x1: Double, x2: Double): Boolean = {
    if (x1 == 0 || x2 == 0) {
      x1 != x2
    } else if (x1 * x2 < 0) {
      true
    } else {
      Math.abs((x1 - x2) / (x1 + x2)) > 1.0e-9
    }
  }

  /**
    * Find the best split on a continuous variable.
    * This is a typical implementation for decision trees: it finds the pivot that maximizes the decrease in impurity.
    * Some splitters might forego this method and implement different ways of choosing a pivot point.
    *
    * @param data  to split
    * @param index of the feature to split on
    * @param minCount minimum number of data points to allow in each of the resulting splits
    * @param randomizePivotLocation whether generate splits by drawing a random value uniformly between the two split points.
    *                               This can improve generalizability, particularly as part of an ensemble.
    * @return the best split of this feature
    */
  def getBestRealSplit[T](
      data: Seq[(Vector[AnyVal], T, Double)],
      calculator: ImpurityCalculator[T],
      index: Int,
      minCount: Int,
      randomizePivotLocation: Boolean = false,
      rng: Random = Random
  ): (Split, Double) = {
    /* Pull out the feature that's considered here and sort by it */
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)
    val features = thinData.map(x => x._1)

    var bestImpurity = Double.MaxValue
    var bestPivot = Double.MinValue

    /* Move the data from the right to the left partition one value at a time */
    calculator.reset()
    (0 until data.size - minCount).foreach { j =>
      val totalImpurity = calculator.add(thinData(j)._2, thinData(j)._3)
      val left = features(j + 1)
      val right = features(j)
      if (totalImpurity < bestImpurity && j + 1 >= minCount && isDifferent(left, right)) {
        bestImpurity = totalImpurity
        /* Try pivots at the midpoints between consecutive member values */
        bestPivot = if (randomizePivotLocation) {
          (left - right) * rng.nextDouble() + right
        } else {
          (left + right) / 2.0
        }
      }
    }
    (new RealSplit(index, bestPivot), bestImpurity)
  }
}
