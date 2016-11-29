package io.citrine.lolo.trees


abstract trait Split {
  def turnLeft(input: Array[AnyVal]): Boolean
}

class RealSplit(index: Int, pivot: Double) extends Split {
  override def turnLeft(input: Array[AnyVal]): Boolean = {
    input(index).asInstanceOf[Double] <= pivot
  }
}

class CategoricalSplit(index: Int, include: Set[Char]) extends Split {
  override def turnLeft(input: Array[AnyVal]): Boolean = {
    include.contains(input(index).asInstanceOf[Char])
  }
}


/**
  * Created by maxhutch on 11/29/16.
  */
object RegressionSplitter {

  def getBestSplit(data: Seq[(Array[AnyVal], Double)]): Split = {
    var bestSplit: Split = null
    var bestVariance = Double.MaxValue

    val totalSum = data.map(d => d._2).sum
    val totalNum = data.size

    /* Try every feature index */
    (0 until data.head._1.size).foreach { index =>
      /* Get the list of feature values */
      val rep = data.head._1(index)

      val (possibleSplit, possibleVariance) = rep match {
        case x: Double => getBestRealSplit(data, totalSum, index)
        case x: Char => getBestCategoricalSplit(data, totalSum, index)
        case x: Any => (null, Double.MaxValue)
      }

      if (possibleVariance < bestVariance) {
        bestVariance = possibleVariance
        bestSplit = possibleSplit
      }

    }
    bestSplit
  }

  def getBestRealSplit(data: Seq[(Array[AnyVal], Double)], totalSum: Double, index: Int): (RealSplit, Double) = {
    val totalNum = data.size
    var leftSum = 0.0
    var bestVariance = Double.MaxValue
    var bestPivot = Double.MinValue
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Double], dat._2)).sortBy(_._1)

    /* Try pivots at the midpoints between consecutive member values */
    (1 until data.size).foreach { j =>
      leftSum = leftSum + thinData(j - 1)._2

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalVariance = - leftSum * leftSum / j - Math.pow(totalSum - leftSum, 2) / (totalNum - j)

      /* Keep track of the best split */
      if (totalVariance < bestVariance) {
        bestVariance = totalVariance
        bestPivot = (thinData(j)._1 + thinData(j - 1)._1) / 2.0
      }
    }
    (new RealSplit(index, bestPivot), bestVariance)
  }

  def getBestCategoricalSplit(data: Seq[(Array[AnyVal], Double)], totalSum: Double, index: Int): (CategoricalSplit, Double) = {
    val totalNum = data.size
    var leftSum = 0.0
    var leftNum = 0
    var bestVariance = Double.MaxValue
    var bestSet = Set.empty[Char]
    val thinData = data.map(dat => (dat._1(index).asInstanceOf[Char], dat._2))
    val categoryAverages: Map[Char, Double] = thinData.groupBy(_._1).mapValues(g => g.map(_._2).sum / g.size)
    val order = categoryAverages.toSeq.sortBy(_._2).map(_._1)


    /* Try pivots at the midpoints between consecutive member values */
    (0 until order.size - 1).foreach { j =>
      val toAdd = thinData.filter(_._1 == order(j))
      leftSum = leftSum + toAdd.map(_._2).sum
      leftNum = leftNum + toAdd.size

      /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
      val totalVariance = - leftSum * leftSum / leftNum - Math.pow(totalSum - leftSum, 2) / (totalNum - leftNum)

      /* Keep track of the best split */
      if (totalVariance < bestVariance) {
        bestVariance = totalVariance
        bestSet = order.slice(0, j).toSet
      }
    }
    (new CategoricalSplit(index, bestSet), bestVariance)
  }

}
