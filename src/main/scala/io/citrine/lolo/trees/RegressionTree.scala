package io.citrine.lolo.trees

import io.citrine.lolo.encoders.CategoricalEncoder

/**
  * Created by maxhutch on 11/28/16.
  */
class RegressionTreeLearner() {

  def train(trainingData: Seq[(Vector[Any], Double)]): RegressionTree = {
    val repInput = trainingData.head._1

    val encoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map{ case (v,i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(trainingData.map(_._1(i))))
      }
    }

    val encodedTraining = trainingData.map(p => (RegressionTree.encodeInput(p._1, encoders), p._2))
    val rootTrainingNode = new RegressionTrainingNode(encodedTraining)
    val importance = rootTrainingNode.getFeatureImportance()
    new RegressionTree(rootTrainingNode.getNode(), encoders, importance.map(_ / importance.sum))
  }

}

class RegressionTree(root: ModelNode[AnyVal], encoders: Seq[Option[CategoricalEncoder[Any]]], importance: Array[Double]) {
  def predict(input: Vector[Any]): Double = {
    root.predict(RegressionTree.encodeInput(input, encoders))
  }

  def getFeatureImportance(): Array[Double] = importance
}

object RegressionTree {
  def encodeInput(input: Vector[Any], encoders: Seq[Option[CategoricalEncoder[Any]]]): Vector[AnyVal] = {
    input.zip(encoders).map{ case (v, e) =>
        e match {
          case Some(x) => x.encode(v)
          case None => v.asInstanceOf[AnyVal]
        }
    }
  }
}

class RegressionTrainingNode (
                             trainingData: Seq[(Vector[AnyVal], Double)],
                             impurityIn: Double = -1.0,
                             remainingDepth: Int = Int.MaxValue
                             )
  extends TrainingNode[AnyVal] (
    trainingData = trainingData,
    impurity = impurityIn,
    remainingDepth = remainingDepth
  ) {

  lazy val split: Split = RegressionSplitter.getBestSplit(trainingData)
  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0) {
    new RegressionTrainingNode(leftTrain, remainingDepth = remainingDepth - 1)
  } else {
    new RegressionTrainingLeaf(leftTrain)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0) {
    new RegressionTrainingNode(rightTrain, remainingDepth = remainingDepth - 1)
  } else {
    new RegressionTrainingLeaf(rightTrain)
  }

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[AnyVal] = {
    new RegressionModelNode(split, leftChild.getNode(), rightChild.getNode())
  }

  override def getFeatureImportance(): Array[Double] = {
    val improvement = getImpurity() - leftChild.getImpurity() - rightChild.getImpurity()
    var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex) = ans(split.getIndex) + improvement
    ans
  }
}

/**
  * Average the training data to make a leaf prediction
  * @param trainingData to train on
  */
class RegressionTrainingLeaf(
                          trainingData: Seq[(Vector[AnyVal], Double)],
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
  def getNode(): ModelNode[AnyVal] = {
    new RegressionLeaf(trainingData.map(_._2).sum / trainingData.size)
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

class RegressionModelNode(split: Split, left: ModelNode[AnyVal], right: ModelNode[AnyVal]) extends ModelNode[AnyVal] {
  /**
  * Just propagate the prediction call through the appropriate child
  * @param input to predict for
  * @return prediction
  */
  override def predict(input: Vector[AnyVal]): Double = {
    if (split.turnLeft(input)) {
      left.predict(input)
    } else {
      right.predict(input)
    }
  }
}

class RegressionLeaf(mean: Double) extends ModelNode[AnyVal] {
  override def predict(input: Vector[AnyVal]): Double = mean
}
