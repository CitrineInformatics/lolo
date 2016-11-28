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
    new SimpleRegressionTree(new SimpleTrainingNode(trainingData).getNode())
  }
}

/**
  * Simple regression tree that wraps a Node's predict method
  * @param root
  */
class SimpleRegressionTree(root: Node) {
  /**
    * Make a prediction from an input vector
    * @param input vector of doubles
    * @return prediction as a double
    */
  def predict(input: Array[Double]): Double = root.predict(input)
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
}

/**
  * Internal node written simply
  * @param trainingData to train the split on
  */
class SimpleTrainingNode(trainingData: Seq[(Array[Double], Double)]) extends TrainingNode {

  /**
    * Wrap some internal state in a lightweight node
    * @return lightweight prediction node
    */
  override def getNode(): Node = {
    new SimpleNode(index, pivot, leftChild.getNode(), rightChild.getNode())
  }

  lazy val (index: Int, pivot: Double) = SimpleTrainingNode.getBestSplit(trainingData)
  lazy val (leftTrain, rightTrain) = trainingData.partition{ case (x, y) => x(index) <= pivot }
  lazy val leftChild = if (leftTrain.size > 1) {
    new SimpleTrainingNode(leftTrain)
  } else {
    new SimpleTrainingLeaf(leftTrain)
  }
  lazy val rightChild = if (rightTrain.size > 1) {
    new SimpleTrainingNode(rightTrain)
  } else {
    new SimpleTrainingLeaf(rightTrain)
  }
}

/**
  * Average the training data to make a leaf prediction
  * @param trainingData to train on
  */
class SimpleTrainingLeaf(trainingData: Seq[(Array[Double], Double)]) extends TrainingNode {
  /**
    * Average the training data
    * @return lightweight prediction node
    */
  def getNode(): Node = {
    new SimpleLeaf(trainingData.map(_._2).sum / trainingData.size)
  }
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
}

/**
  * Just holds the mean
  * @param mean prediction of the mode
  */
class SimpleLeaf(mean: Double) extends Node {
  override def predict(input: Array[Double]): Double = mean
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
    val splits = (0 until data.head._1.size).foreach { index =>
      /* Get the list of feature values */
      val members = data.map(_._1(index)).distinct.sorted
      /* Try pivots at the midpoints between consecutive member values */
      (0 until members.size - 1).foreach { j =>
        val pivot = (members(j) + members(j + 1))/2.0

        /* Use foreach w/var so we can walk through the data once */
        var leftMean = 0.0
        var leftSqar = 0.0
        var leftNum: Int = 0
        var rightMean = 0.0
        var rightSqar = 0.0
        var rightNum: Int = 0
        (0 until data.size).foreach{ k =>
          val x = data(k)._1(index)
          if (x <= pivot) {
            leftMean = leftMean + x
            leftSqar = leftSqar + x*x
            leftNum  = leftNum + 1
          } else {
            rightMean = rightMean + x
            rightSqar = rightSqar + x*x
            rightNum  = rightNum + 1
          }
        }
        val totalVariance = (leftSqar - leftMean*leftMean / leftNum) + (rightSqar - rightMean*rightMean / rightNum)

        /* Keep track of the best split */
        if (totalVariance < bestVariance){
          bestVariance = totalVariance
          bestPivot = pivot
          bestIndex = index
        }
      }
    }
    (bestIndex, bestPivot)
  }
}