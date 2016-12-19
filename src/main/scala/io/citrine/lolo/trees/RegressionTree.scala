package io.citrine.lolo.trees

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.results.{PredictionResult, TrainingResult, hasFeatureImportance}
import io.citrine.lolo.trees.splits.{NoSplit, RegressionSplitter, Split}
import io.citrine.lolo.{Learner, Model}

/**
  * Learner for regression trees
  *
  * Created by maxhutch on 11/28/16.
  *
  * @param numFeatures to randomly select from at each split (default: all)
  * @param maxDepth    to grow the tree to
  * @param leafLearner learner to train the leaves with
  */
class RegressionTreeLearner(
                             numFeatures: Int = -1,
                             maxDepth: Int = 30,
                             leafLearner: Option[Learner] = None
                           ) extends Learner {

  override def setHypers(moreHypers: Map[String, Any]): this.type = {
    hypers = hypers ++ moreHypers
    myLeafLearner.setHypers(moreHypers)
    this
  }

  /** Hyperparameters */
  var hypers: Map[String, Any] = Map("minLeafInstances" -> 1)

  /** Learner to use for training the leaves */
  val myLeafLearner = leafLearner.getOrElse(new GuessTheMeanLearner())

  /**
    * Train the tree by recursively partitioning (splitting) the training data on a single feature
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a RegressionTree
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): RegressionTreeTrainingResult = {
    if (!trainingData.head._2.isInstanceOf[Double]) {
      throw new IllegalArgumentException(s"Tried to train regression on non-double labels, e.g.: ${trainingData.head._2}")
    }
    assert(trainingData.size > 4, s"We need to have at least 4 rows, only ${trainingData.size} given")

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
    }.filter(_._3 > 0).toVector

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    /* The tree is built of training nodes */
    val (split, delta) = RegressionSplitter.getBestSplit(finalTraining, numFeaturesActual, hypers("minLeafInstances").asInstanceOf[Int])
    val rootTrainingNode = if (split.isInstanceOf[NoSplit]) {
      new RegressionTrainingLeaf(finalTraining)
    } else {
      new RegressionTrainingNode(
        finalTraining,
        split,
        delta,
        numFeaturesActual,
        minLeafInstances = hypers("minLeafInstances").asInstanceOf[Int],
        remainingDepth = maxDepth - 1)
    }

    /* Wrap them up in a regression tree */
    new RegressionTreeTrainingResult(rootTrainingNode, encoders)
  }

  class RegressionTrainingNode(
                                trainingData: Seq[(Vector[AnyVal], Double, Double)],
                                split: Split,
                                deltaImpurity: Double,
                                numFeatures: Int,
                                minLeafInstances: Int,
                                remainingDepth: Int
                              )
    extends TrainingNode(
      trainingData = trainingData,
      remainingDepth = remainingDepth
    ) {

    // val (split: Split, deltaImpurity: Double) = RegressionSplitter.getBestSplit(trainingData, numFeatures)
    assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
    assert(!split.isInstanceOf[NoSplit], s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}")

    lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
    assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}")

    lazy val leftChild = if (leftTrain.size >= 2 * minLeafInstances && remainingDepth > 0 && leftTrain.exists(_._2 != leftTrain.head._2)) {
      lazy val (leftSplit, leftDelta) = RegressionSplitter.getBestSplit(leftTrain, numFeatures, minLeafInstances)
      if (!leftSplit.isInstanceOf[NoSplit]) {
        new RegressionTrainingNode(leftTrain, leftSplit, leftDelta, numFeatures, minLeafInstances, remainingDepth - 1)
      } else {
        new RegressionTrainingLeaf(leftTrain)
      }
    } else {
      new RegressionTrainingLeaf(leftTrain)
    }

    lazy val rightChild = if (rightTrain.size >= 2 * minLeafInstances && remainingDepth > 0 && rightTrain.exists(_._2 != rightTrain.head._2)) {
      lazy val (rightSplit, rightDelta) = RegressionSplitter.getBestSplit(rightTrain, numFeatures, minLeafInstances)
      if (!rightSplit.isInstanceOf[NoSplit]) {
        new RegressionTrainingNode(rightTrain, rightSplit, rightDelta, numFeatures, minLeafInstances, remainingDepth - 1)
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
    override def getNode(): Model[PredictionResult[Double]] = {
      new InternalModelNode[PredictionResult[Double]](split, leftChild.getNode(), rightChild.getNode())
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
    def getNode(): Model[PredictionResult[Double]] = {
        myLeafLearner.train(trainingData).getModel().asInstanceOf[Model[PredictionResult[Double]]]
    }

    override def getFeatureImportance(): Array[Double] = Array.fill(trainingData.head._1.size)(0.0)
  }

}

class RegressionTreeTrainingResult(
                                    rootTrainingNode: TrainingNode[AnyVal, Double],
                                    encoders: Seq[Option[CategoricalEncoder[Any]]]
                                  ) extends TrainingResult with hasFeatureImportance {
  lazy val model = new RegressionTree(rootTrainingNode.getNode(), encoders)
  lazy val importance = rootTrainingNode.getFeatureImportance()
  lazy val importanceNormalized = importance.map(_ / importance.sum)

  override def getModel(): RegressionTree = model

  /**
    * Return the pre-computed importances
    *
    * @return feature importances as an array of doubles
    */
  override def getFeatureImportance(): Array[Double] = importanceNormalized
}

/**
  * Container holding a model node, encoders, and the feature importances
  *
  * @param root     of the tree
  * @param encoders for categorical variables
  */
class RegressionTree(
                      root: Model[PredictionResult[Double]],
                      encoders: Seq[Option[CategoricalEncoder[Any]]]
                    ) extends Model[RegressionTreeResult] {
  /**
    * Apply the model by calling predict and wrapping the results
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes only the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): RegressionTreeResult = {
    new RegressionTreeResult(
      inputs.map(inp => root.transform(Seq(RegressionTree.encodeInput(inp, encoders)))),
      inputs.head
    )
  }
}

/**
  * Simple wrapper around a sequence of predictions
  *
  * @param predictions sequence of predictions
  */
class RegressionTreeResult(predictions: Seq[PredictionResult[Double]], repInput: Seq[Any]) extends PredictionResult[Double] {
  /**
    * Get the predictions
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = predictions.map(_.getExpected().head)

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = {
    if (!predictions.head.getGradient().isDefined) {
      return None
    }
    Some(predictions.map(_.getGradient().get.head))
  }
}

/** Companion object with common utilities */
object RegressionTree {
  /**
    * Apply a sequence of encoders to transform categorical variables into chars
    *
    * @param input    to encode
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


