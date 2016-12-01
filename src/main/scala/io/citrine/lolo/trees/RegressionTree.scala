package io.citrine.lolo.trees

import io.citrine.lolo.{Learner, Model, PredictionResult}
import io.citrine.lolo.encoders.CategoricalEncoder

/**
  * Created by maxhutch on 11/28/16.
  */
class RegressionTreeLearner(numFeatures: Int = -1, maxDepth: Int = 30) extends Learner {

  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): RegressionTree = {
    if (!trainingData.head._2.isInstanceOf[Double]) {
      throw new IllegalArgumentException(s"Tried to train regression on non-double labels, e.g.: ${trainingData.head._2}")
    }
    val repInput = trainingData.head._1

    val encoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map{ case (v,i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(trainingData.map(_._1(i))))
      }
    }

    val encodedTraining = trainingData.map(p => (RegressionTree.encodeInput(p._1, encoders), p._2))
    val finalTraining = encodedTraining.zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0))).map{ case ((f, l), w) =>
      (f, l.asInstanceOf[Double], w)
    }.filter(_._3 > 0)

    val numFeaturesActual = if (numFeatures > 0){
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    val rootTrainingNode = new RegressionTrainingNode(finalTraining, numFeaturesActual, remainingDepth = maxDepth)
    val rootModelNode = rootTrainingNode.getNode()
    val importance = rootTrainingNode.getFeatureImportance()
    new RegressionTree(rootModelNode, encoders, importance.map(_ / importance.sum))
  }

}

class RegressionTree(
                      root: ModelNode[AnyVal],
                      encoders: Seq[Option[CategoricalEncoder[Any]]],
                      importance: Array[Double]
                    ) extends Model {
  def predict(input: Vector[Any]): Double = {
    root.predict(RegressionTree.encodeInput(input, encoders))
  }

  def predict(inputs: Seq[Vector[Any]]): Seq[Double] = {
    inputs.map(predict)
  }

  override def transform(inputs: Seq[Vector[Any]]): PredictionResult = {
    new RegressionTreeResult(inputs.map(predict))
  }

  override def getFeatureImportance(): Array[Double] = importance
}

class RegressionTreeResult(predictions: Seq[Double]) extends PredictionResult {
  override def getExpected(): Seq[Any] = predictions
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
                               trainingData: Seq[(Vector[AnyVal], Double, Double)],
                               numFeatures: Int,
                               impurityIn: Double = -1.0,
                               remainingDepth: Int = Int.MaxValue
                             )
  extends TrainingNode[AnyVal] (
    trainingData = trainingData,
    impurity = impurityIn,
    remainingDepth = remainingDepth
  ) {

  val split: Split = RegressionSplitter.getBestSplit(trainingData, numFeatures)
  // assert(split != null, s"Null split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  // assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split (${split}) was external for: ${trainingData.map(t => (t._1(split.getIndex()), t._2, t._3))}")
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0 && leftTrain.exists(_._2 != leftTrain.head._2)) {
    val tryNode = new RegressionTrainingNode(leftTrain, numFeatures, remainingDepth = remainingDepth - 1)
    if (tryNode.split != null) {
      tryNode
    } else {
      new RegressionTrainingLeaf(leftTrain)
    }
  } else {
    new RegressionTrainingLeaf(leftTrain)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0 && rightTrain.exists(_._2 != rightTrain.head._2)) {
    val tryNode = new RegressionTrainingNode(rightTrain, numFeatures, remainingDepth = remainingDepth - 1)
    if (tryNode.split != null){
      tryNode
    } else {
      new RegressionTrainingLeaf(rightTrain)
    }
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
                          trainingData: Seq[(Vector[AnyVal], Double, Double)],
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
