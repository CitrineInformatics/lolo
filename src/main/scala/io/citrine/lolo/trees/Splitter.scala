package io.citrine.lolo.trees

import scala.util.Random

/**
  * Splits are used by decision trees to partition the input space
  */
abstract trait Split {
  /**
    * Take the left branch in the binary split?
    * @param input vector of any type
    * @return true if input takes the left split
    */
  def turnLeft(input: Vector[AnyVal]): Boolean

  /**
    * Get the index of the input vector that is used to pick this split
    * @return index of the input vector used by this split
    */
  def getIndex(): Int
}

/**
  * Split based on a real value in the index position
  * @param index position of the real value to inspect
  * @param pivot value at or below which to take the left split
  */
class RealSplit(index: Int, pivot: Double) extends Split {

  /**
    * If the value is at or less than the pivot, turn left
    *
    * @param input vector of any type
    * @return true if input takes the left split
    */
  override def turnLeft(input: Vector[AnyVal]): Boolean = {
    input(index).asInstanceOf[Double] <= pivot
  }

  /**
    * ${inherit_doc}
    * @return index of the input vector used by this split
    */
  override def getIndex: Int = index

  /**
    * Pretty print
    * @return debug string
    */
  override def toString: String = s"Split index ${index} @ ${pivot}"
}

class CategoricalSplit(index: Int, include: Set[Char]) extends Split {
  override def turnLeft(input: Vector[AnyVal]): Boolean = {
    include.contains(input(index).asInstanceOf[Char])
  }
  override def getIndex: Int = index
}

/**
  * Created by maxhutch on 11/29/16.
  */
object RegressionSplitter {

  def getBestSplit(data: Seq[(Vector[AnyVal], Double, Double)], numFeatures: Int): Split = {
    var bestSplit: Split = null
    var bestVariance = Double.MaxValue

    val totalSum = data.map(d => d._2 * d._3).sum
    val totalWeight = data.map(d => d._3).sum

    /* Try every feature index */
    val featureIndices: Seq[Int] = data.head._1.indices
    Random.shuffle(featureIndices).take(numFeatures).foreach { index =>
      /* Get the list of feature values */
      val rep = data.head._1(index)

      val (possibleSplit, possibleVariance) = rep match {
        case x: Double => getBestRealSplit(data, totalSum, totalWeight, index)
        case x: Char => getBestCategoricalSplit(data, totalSum, totalWeight, index)
        case x: Any => throw new IllegalArgumentException
      }

      if (possibleVariance < bestVariance) {
        bestVariance = possibleVariance
        bestSplit = possibleSplit
      }
    }
    bestSplit
  }

  def getBestRealSplit(data: Seq[(Vector[AnyVal], Double, Double)], totalSum: Double, totalWeight: Double, index: Int): (RealSplit, Double) = {
    var leftSum = 0.0
    var leftWeight = 0.0
    var bestVariance = Double.MaxValue
    var bestPivot = Double.MinValue
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2, dat._3)).sortBy(_._1)

    /* Try pivots at the midpoints between consecutive member values */
    (1 until data.size).foreach { j =>
      leftSum = leftSum + thinData(j - 1)._2 * thinData(j - 1)._3
      leftWeight = leftWeight + thinData(j - 1)._3

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalVariance = -leftSum * leftSum / leftWeight - Math.pow(totalSum - leftSum, 2) / (totalWeight - leftWeight)
      /*
        val totalVariance = E[squared_left] - (E[left])^2 + E[squared_right] - (E[right])^2
        val totalVariance = (leftSquareSum/leftNum - (leftSum / leftNum)^2) + (rightSquareSum/rightNum - (rightSum / rightNum)^2)
        val totalVariance = (leftSquareSum/leftNum - (leftSum / leftNum)^2) + ((totalSquareSum - leftSquareSum)/(totalNum - leftNum) - ((totalSum - leftSum) / (totalNum - leftNum))^2)
        val totalVariance = leftSquareSum/leftNum - (leftSum / leftNum)^2 + (totalSquareSum - leftSquareSum)/(totalNum - leftNum) - ((totalSum - leftSum) / (totalNum - leftNum))^2

       */

      /* Keep track of the best split
         It is really important for performance to keep these checks together so
         1) there is only one branch and
         2) it is usually false
       */
      if (totalVariance < bestVariance && thinData(j)._1 > thinData(j-1)._1 + 1.0e-9) {
        bestVariance = totalVariance
        bestPivot = thinData(j - 1)._1 // (thinData(j)._1 + thinData(j - 1)._1) / 2.0
      }
    }
    (new RealSplit(index, bestPivot), bestVariance)
  }

  /**
    * Get find the best categorical splitter.
    * @param data
    * @param totalSum
    * @param totalWeight
    * @param index
    * @return
    */
  def getBestCategoricalSplit(data: Seq[(Vector[AnyVal], Double, Double)], totalSum: Double, totalWeight: Double, index: Int): (CategoricalSplit, Double) = {
    var leftSum = 0.0
    var leftWeight = 0.0
    var bestVariance = Double.MaxValue
    var bestSet = Set.empty[Char]
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val groupedData = thinData.groupBy(_._1)
    val categoryAverages: Map[Char, Double] = groupedData.mapValues(g => g.map(v => v._2 * v._3).sum / g.map(_._3).sum)
    val orderedNames = categoryAverages.toSeq.sortBy(_._2).map(_._1)


    /* Try pivots at the midpoints between consecutive member values */
    (0 until orderedNames.size - 1).foreach { j =>
      val toAdd = thinData.filter(_._1 == orderedNames(j))
      leftSum = leftSum + toAdd.map(d => d._2 * d._3).sum
      leftWeight = leftWeight + toAdd.map(d => d._3).sum

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalVariance = - leftSum * leftSum / leftWeight - Math.pow(totalSum - leftSum, 2) / (totalWeight - leftWeight)

      /* Keep track of the best split */
      if (totalVariance < bestVariance) {
        bestVariance = totalVariance
        bestSet = orderedNames.slice(0, j + 1).toSet
      }
    }
    (new CategoricalSplit(index, bestSet), bestVariance)
  }

}
