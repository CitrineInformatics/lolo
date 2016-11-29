package io.citrine.lolo.trees

/**
  * Created by maxhutch on 11/28/16.
  */
class SimpleRegressionTreeLearner {
  /**
    * Train a simple regression tree
    * @param trainingData to train on
    * @return a simple regression tree
    */
  def train(trainingData: Seq[(Vector[Double], Double)]): SimpleRegressionTree = {
    val trainingRoot = new SimpleTrainingNode(trainingData, remainingDepth = 30)
    val importances = trainingRoot.getFeatureImportance()
    new SimpleRegressionTree(trainingRoot.getNode(), importances.map(_ / importances.sum))
  }
}

/**
  * Simple regression tree that wraps a ModelNode's predict method
  * @param root
  */
class SimpleRegressionTree(root: ModelNode[Double], importances: Array[Double]) {
  /**
    * Make a prediction from an input vector
    * @param input vector of doubles
    * @return prediction as a double
    */
  def predict(input: Vector[Double]): Double = root.predict(input)

  def getFeatureImportance(): Array[Double] = importances

  def getRoot(): ModelNode[Double] = root
}

/**
  * Internal node written simply
  * @param trainingData to train the split on
  */
class SimpleTrainingNode(
                          trainingData: Seq[(Vector[Double], Double)],
                          impurityIn: Double = -1.0,
                          remainingDepth: Int = Int.MaxValue
                        ) extends TrainingNode(
  trainingData = trainingData,
  impurity = impurityIn,
  remainingDepth = remainingDepth
) {

  /**
    * Wrap some internal state in a lightweight node
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[Double] = {
    new SimpleModelNode(index, pivot, leftChild.getNode(), rightChild.getNode())
  }

  lazy val (index: Int, pivot: Double) = SimpleTrainingNode.getBestSplit(trainingData)
  lazy val (leftTrain, rightTrain) = trainingData.partition{ case (x, y) => x(index) <= pivot }
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0) {
    new SimpleTrainingNode(leftTrain, remainingDepth = remainingDepth - 1)
  } else {
    new SimpleTrainingLeaf(leftTrain)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0) {
    new SimpleTrainingNode(rightTrain, remainingDepth = remainingDepth - 1)
  } else {
    new SimpleTrainingLeaf(rightTrain)
  }

  override def getFeatureImportance(): Array[Double] = {
    val improvement = getImpurity() - leftChild.getImpurity() - rightChild.getImpurity()
    var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(index) = ans(index) + improvement
    ans
  }

  override def toString(): String = {
    s"Split feature ${index} at ${pivot}"
  }
}

/**
  * Average the training data to make a leaf prediction
  * @param trainingData to train on
  */
class SimpleTrainingLeaf(
                          trainingData: Seq[(Vector[Double], Double)],
                          impurityIn: Double = -1.0
                        ) extends TrainingNode(
  trainingData = trainingData,
  impurity = impurityIn,
  remainingDepth = 0
) {
  /**
    * Average the training data
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[Double] = {
    new SimpleLeaf(trainingData.map(_._2).sum / trainingData.size)
  }

  if (impurity < 0.0) {
    var sum = 0.0
    var sq = 0.0
    trainingData.foreach{ case (x, y) =>
        sum = sum + y
        sq = sq + y * y
    }
    impurity = sq - sum * sum / trainingData.size
  }

  override def getImpurity(): Double = impurity

  override def getFeatureImportance(): Array[Double] = Array.fill(trainingData.head._1.size)(0.0)
}


/**
  * Internal node
  * @param index to split on
  * @param pivot to split at
  * @param left child node
  * @param right child node
  */
class SimpleModelNode(index: Int, pivot: Double, left: ModelNode[Double], right: ModelNode[Double]) extends ModelNode[Double] {
  /**
    * Just propagate the prediction call through the appropriate child
    * @param input to predict for
    * @return prediction
    */
  override def predict(input: Vector[Double]): Double = {
    if (input(index) <= pivot) {
      left.predict(input)
    } else {
      right.predict(input)
    }
  }

  override def toString(): String = {
    s"If (feature ${index} <= ${pivot})\n" + left.toString + s"Else (feature ${index} > ${pivot}\n" + right.toString
  }
}

/**
  * Just holds the mean
  * @param mean prediction of the mode
  */
class SimpleLeaf(mean: Double) extends ModelNode[Double] {
  override def predict(input: Vector[Double]): Double = mean

  override def toString(): String = s"  Predict: ${mean}\n"
}

/**
  * Companion object to hold helper functions
  */
object SimpleTrainingNode {
  /**
    * Get the best split
    * @param data to test splits on
    * @return (index, pivot) tuple
    */
  def getBestSplit(data: Seq[(Vector[Double], Double)]): (Int, Double) = {
    var bestPivot: Double = data.head._1(0)
    var bestVariance = Double.MaxValue
    var bestIndex = -1

    val totalSum = data.map(d => d._2).sum
    val totalNum = data.size

    /* Try every feature index */
    (0 until data.head._1.size).foreach { index =>
      /* Get the list of feature values */
      val thinData = data.map(dat => (dat._1(index), dat._2)).sortBy(_._1)

      var leftSum = 0.0

      /* Try pivots at the midpoints between consecutive member values */
      (1 until thinData.size).foreach { j =>
        leftSum = leftSum + thinData(j-1)._2

        /* This is just relative, so we can subtract off the sum of the squares, data.map(Math.pow(_._2, 2)) */
        val totalVariance = - leftSum * leftSum / j - Math.pow(totalSum - leftSum, 2) / (totalNum - j)

        /* Keep track of the best split */
        if (totalVariance < bestVariance){
          bestVariance = totalVariance
          bestPivot = (thinData(j)._1 + thinData(j-1)._1)/2.0
          bestIndex = index
        }
      }
    }
    (bestIndex, bestPivot)
  }
}