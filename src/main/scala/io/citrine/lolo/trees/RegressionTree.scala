package io.citrine.lolo.trees

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.trees.splits.{NoSplit, RegressionSplitter, Split}
import io.citrine.lolo.{Learner, Model, PredictionResult}

/**
  * Learner for regression trees
  *
  * Created by maxhutch on 11/28/16.
  *
  * @param numFeatures to randomly select from at each split (default: all)
  * @param maxDepth to grow the tree to
  */
class RegressionTreeLearner(numFeatures: Int = -1, maxDepth: Int = 30) extends Learner {

  /**
    * Train the tree by recursively partitioning (splitting) the training data on a single feature
    * @param trainingData to train on
    * @param weights for the training rows, if applicable
    * @return a RegressionTree
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): RegressionTree = {
    if (!trainingData.head._2.isInstanceOf[Double]) {
      throw new IllegalArgumentException(s"Tried to train regression on non-double labels, e.g.: ${trainingData.head._2}")
    }
    val repInput = trainingData.head._1

    /* Create encoders for any categorical features */
    val encoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(trainingData.map(_._1(i))))
      }
    }

    /* Encode the training data */
    val encodedTraining = trainingData.map(p => (RegressionTree.encodeInput(p._1, encoders), p._2))

    /* Add the weights to the (features, label) tuples and remove any with zero weight */
    val finalTraining = encodedTraining.zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0))).map { case ((f, l), w) =>
      (f, l.asInstanceOf[Double], w)
    }.filter(_._3 > 0)

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    /* The tree is built of training nodes */
    val rootTrainingNode = new RegressionTrainingNode(finalTraining, numFeaturesActual, remainingDepth = maxDepth)

    /* Grab a prediction node.  The partitioning happens here */
    val rootModelNode = rootTrainingNode.getNode()

    /* Grab the feature importances */
    val importance = rootTrainingNode.getFeatureImportance()

    /* Wrap them up in a regression tree */
    new RegressionTree(rootModelNode, encoders, importance.map(_ / importance.sum))
  }
}

/**
  * Container holding a model node, encoders, and the feature importances
  * @param root of the tree
  * @param encoders for categorical variables
  * @param importance feature importances
  */
class RegressionTree(
                      root: ModelNode[AnyVal, Double],
                      encoders: Seq[Option[CategoricalEncoder[Any]]],
                      importance: Array[Double]
                    ) extends Model {

  /**
    * Make a regression prediction
    * @param input features
    * @return predicted response as a double
    */
  def predict(input: Vector[Any]): Double = {
    root.predict(RegressionTree.encodeInput(input, encoders))
  }

  /**
    * Make many regression predictions
    * @param inputs to predict
    * @return sequence of predictions
    */
  def predict(inputs: Seq[Vector[Any]]): Seq[Double] = {
    inputs.map(predict)
  }

  /**
    * Apply the model by calling predict and wrapping the results
    * @param inputs to apply the model to
    * @return a predictionresult which includes only the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): PredictionResult = {
    new RegressionTreeResult(inputs.map(predict))
  }

  /**
    * Return the pre-computed importances
    * @return feature importances as an array of doubles
    */
  override def getFeatureImportance(): Array[Double] = importance
}

/**
  * Simple wrapper around a sequence of predictions
  * @param predictions sequence of predictions
  */
class RegressionTreeResult(predictions: Seq[Double]) extends PredictionResult {
  /**
    * Get the predictions
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = predictions
}

/** Companion object with common utilities */
object RegressionTree {
  /**
    * Apply a sequence of encoders to transform categorical variables into chars
    * @param input to encode
    * @param encoders sequence of encoders
    * @return input with categoricals encoded as chars
    */
  def encodeInput(input: Vector[Any], encoders: Seq[Option[CategoricalEncoder[Any]]]): Vector[AnyVal] = {
    input.zip(encoders).map { case (v, e) =>
      e match {
        case Some(x) => x.encode(v)
        case None => v.asInstanceOf[AnyVal]
      }
    }
  }
}

class RegressionTrainingNode(
                              trainingData: Seq[(Vector[AnyVal], Double, Double)],
                              numFeatures: Int,
                              remainingDepth: Int = Int.MaxValue
                            )
  extends TrainingNode(
    trainingData = trainingData,
    remainingDepth = remainingDepth
  ) {

  val (split: Split, deltaImpurity: Double) = RegressionSplitter.getBestSplit(trainingData, numFeatures)
  // assert(split != null, s"Null split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  // assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split (${split}) was external for: ${trainingData.map(t => (t._1(split.getIndex()), t._2, t._3))}")
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0 && leftTrain.exists(_._2 != leftTrain.head._2)) {
    val tryNode = new RegressionTrainingNode(leftTrain, numFeatures, remainingDepth = remainingDepth - 1)
    if (tryNode.split != NoSplit) {
      tryNode
    } else {
      new RegressionTrainingLeaf(leftTrain)
    }
  } else {
    new RegressionTrainingLeaf(leftTrain)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0 && rightTrain.exists(_._2 != rightTrain.head._2)) {
    val tryNode = new RegressionTrainingNode(rightTrain, numFeatures, remainingDepth = remainingDepth - 1)
    if (tryNode.split != NoSplit) {
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
  override def getNode(): ModelNode[AnyVal, Double] = {
    new InternalModelNode(split, leftChild.getNode(), rightChild.getNode())
  }

  override def getFeatureImportance(): Array[Double] = {
    val improvement = deltaImpurity
    var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex) = ans(split.getIndex) + improvement
    ans
  }
}

/**
  * Average the training data to make a leaf prediction
  *
  * @param trainingData to train on
  */
class RegressionTrainingLeaf(
                              trainingData: Seq[(Vector[AnyVal], Double, Double)]
                            ) extends TrainingNode(
  trainingData = trainingData,
  remainingDepth = 0
) {
  /**
    * Average the training data
    *
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[AnyVal, Double] = {
    new RegressionLeaf(trainingData.map(_._2).sum / trainingData.size)
  }

  override def getFeatureImportance(): Array[Double] = Array.fill(trainingData.head._1.size)(0.0)
}

class RegressionLeaf(mean: Double) extends ModelNode[AnyVal, Double] {
  override def predict(input: Vector[AnyVal]): Double = mean
}
