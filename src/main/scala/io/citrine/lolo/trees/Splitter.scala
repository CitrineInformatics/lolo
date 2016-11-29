package io.citrine.lolo.trees


abstract trait Split {
  def turnLeft(input: Vector[AnyVal]): Boolean
  def getIndex(): Int
}

class RealSplit(index: Int, pivot: Double) extends Split {
  override def turnLeft(input: Vector[AnyVal]): Boolean = {
    input(index).asInstanceOf[Double] <= pivot
  }
  override def getIndex: Int = index
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

  def getBestSplit(data: Seq[(Vector[AnyVal], Double, Double)]): Split = {
    var bestSplit: Split = null
    var bestVariance = Double.MaxValue

    val totalSum = data.map(d => d._2 * d._3).sum
    val totalWeight = data.map(d => d._3).sum

    /* Try every feature index */
    (0 until data.head._1.size).foreach { index =>
      /* Get the list of feature values */
      val rep = data.head._1(index)

      val (possibleSplit, possibleVariance) = rep match {
        case x: Double => getBestRealSplit(data, totalSum, totalWeight, index)
        case x: Char => getBestCategoricalSplit(data, totalSum, totalWeight, index)
        case x: Any => (null, Double.MaxValue)
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
      val totalVariance = - leftSum * leftSum / leftWeight - Math.pow(totalSum - leftSum, 2) / (totalWeight - leftWeight)

      /* Keep track of the best split */
      if (totalVariance < bestVariance) {
        bestVariance = totalVariance
        bestPivot = (thinData(j)._1 + thinData(j - 1)._1) / 2.0
      }
    }
    (new RealSplit(index, bestPivot), bestVariance)
  }

  def getBestCategoricalSplit(data: Seq[(Vector[AnyVal], Double, Double)], totalSum: Double, totalWeight: Double, index: Int): (CategoricalSplit, Double) = {
    var leftSum = 0.0
    var leftWeight = 0.0
    var bestVariance = Double.MaxValue
    var bestSet = Set.empty[Char]
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2, dat._3))
    val categoryAverages: Map[Char, Double] = thinData.groupBy(_._1).mapValues(g => g.map(_._2).sum / g.size)
    val order = categoryAverages.toSeq.sortBy(_._2).map(_._1)


    /* Try pivots at the midpoints between consecutive member values */
    (0 until order.size - 1).foreach { j =>
      val toAdd = thinData.filter(_._1 == order(j))
      leftSum = leftSum + toAdd.map(d => d._2 * d._3).sum
      leftWeight = leftWeight + toAdd.map(d => d._3).sum

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalVariance = - leftSum * leftSum / leftWeight - Math.pow(totalSum - leftSum, 2) / (totalWeight - leftWeight)

      /* Keep track of the best split */
      if (totalVariance < bestVariance) {
        bestVariance = totalVariance
        bestSet = order.slice(0, j + 1).toSet
      }
    }
    (new CategoricalSplit(index, bestSet), bestVariance)
  }

}
