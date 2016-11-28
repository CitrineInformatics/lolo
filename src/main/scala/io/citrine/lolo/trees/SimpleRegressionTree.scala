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
  def train(trainingData: Seq[(Array[Double], Double)]): SimpleRegressionTree = {
    val trainingRoot = new SimpleTrainingNode(trainingData, remainingDepth = 30)
    val importances = trainingRoot.getFeatureImportance()
    new SimpleRegressionTree(trainingRoot.getNode(), importances.map(_ / importances.sum))
  }
}

/**
  * Simple regression tree that wraps a Node's predict method
  * @param root
  */
class SimpleRegressionTree(root: Node, importances: Array[Double]) {
  /**
    * Make a prediction from an input vector
    * @param input vector of doubles
    * @return prediction as a double
    */
  def predict(input: Array[Double]): Double = root.predict(input)

  def getFeatureImportance(): Array[Double] = importances

  def getRoot(): Node = root
}

/**
  * Trait to provide getNode interface for internal and leaf training nodes
  */
abstract trait TrainingNode {
  /**
    * Get the lightweight prediction node for the output tree
    * @return lightweight prediction node
    */
  def getNode(): Node

  def getImpurity(): Double

  def getFeatureImportance(): Array[Double]
}

/**
  * Internal node written simply
  * @param trainingData to train the split on
  */
class SimpleTrainingNode(
                          trainingData: Seq[(Array[Double], Double)],
                          var impurity: Double = -1.0,
                          remainingDepth: Int = Int.MaxValue
                        ) extends TrainingNode {

  /**
    * Wrap some internal state in a lightweight node
    * @return lightweight prediction node
    */
  override def getNode(): Node = {
    new SimpleNode(index, pivot, leftChild.getNode(), rightChild.getNode())
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
class SimpleTrainingLeaf(trainingData: Seq[(Array[Double], Double)], var impurity: Double = -1.0) extends TrainingNode {
  /**
    * Average the training data
    * @return lightweight prediction node
    */
  def getNode(): Node = {
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
  * Trait to hold prediction interface
  */
abstract trait Node {
  def predict(input: Array[Double]): Double
}

/**
  * Internal node
  * @param index to split on
  * @param pivot to split at
  * @param left child node
  * @param right child node
  */
class SimpleNode(index: Int, pivot: Double, left: Node, right: Node) extends Node {
  /**
    * Just propagate the prediction call through the appropriate child
    * @param input to predict for
    * @return prediction
    */
  override def predict(input: Array[Double]): Double = {
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
class SimpleLeaf(mean: Double) extends Node {
  override def predict(input: Array[Double]): Double = mean

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
  def getBestSplit(data: Seq[(Array[Double], Double)]): (Int, Double) = {
    var bestPivot = Double.MinValue
    var bestVariance = Double.MaxValue
    var bestIndex = -1
    /* Try every feature index */
    (0 until data.head._1.size).foreach { index =>
      /* Get the list of feature values */
      val thinData = data.map(dat => (dat._1(index), dat._2)).sortBy(_._1)

      var rightSum = thinData.map(d => d._2).sum
      var rightSq = thinData.map(d => d._2 * d._2).sum
      var rightNum = thinData.size
      var leftSum = 0.0
      var leftSq = 0.0

      /* Try pivots at the midpoints between consecutive member values */
      (0 until thinData.size - 1).foreach { j =>
        leftSum = leftSum + thinData(j)._2
        rightSum = rightSum - thinData(j)._2
        leftSq = leftSq + thinData(j)._2 * thinData(j)._2
        rightSq = rightSq - thinData(j)._2 * thinData(j)._2
        rightNum = rightNum - 1

        val totalVariance = (leftSq - leftSum * leftSum / (thinData.size - rightNum)) + (rightSq - rightSum * rightSum / rightNum)

        /* Keep track of the best split */
        if (totalVariance < bestVariance){
          bestVariance = totalVariance
          bestPivot = (thinData(j)._1 + thinData(j+1)._1)/2.0
          bestIndex = index
        }
      }
    }
    (bestIndex, bestPivot)
  }
}