package io.citrine.lolo.trees.regression

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.splits.{NoSplit, RegressionSplitter, Splitter}
import io.citrine.lolo.trees.splits.{NoSplit, RegressionSplitter}
import io.citrine.lolo.trees.{ModelNode, TrainingNode, TreeMeta}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

import scala.util.Random

/**
  * Learner for regression trees
  *
  * Created by maxhutch on 11/28/16.
  *
  * @param numFeatures to randomly select from at each split (default: all)
  * @param maxDepth    to grow the tree to
  * @param leafLearner learner to train the leaves with
  * @param minLeafInstances minimum number of instances per leaf
  */
case class RegressionTreeLearner(
                                  numFeatures: Int = -1,
                                  maxDepth: Int = 30,
                                  minLeafInstances: Int = 1,
                                  leafLearner: Option[Learner] = None,
                                  splitter: Splitter[Double] = RegressionSplitter(),
                                  rng: Random = Random
                                ) extends Learner {
  /** Learner to use for training the leaves */
  @transient private lazy val myLeafLearner = leafLearner.getOrElse(GuessTheMeanLearner(rng = rng))

  /**
    * Train the tree by recursively partitioning (splitting) the training data on a single feature
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a RegressionTree
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): RegressionTreeTrainingResult = {
    require(trainingData.nonEmpty, s"The input training data was empty")
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
    val encodedTraining = trainingData.map(p => (CategoricalEncoder.encodeInput(p._1, encoders), p._2))

    /* Add the weights to the (features, label) tuples and remove any with zero weight */
    val finalTraining = encodedTraining.zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0))).map { case ((f, l), w) =>
      (f, l.asInstanceOf[Double], w)
    }.filter(_._3 > 0).toVector

    require(
      finalTraining.size >= 4,
      s"We need to have at least 4 rows with non-zero weights, only ${finalTraining.size} given"
    )

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    /* The tree is built of training nodes */
    val (split, delta) = splitter.getBestSplit(finalTraining, numFeaturesActual, minLeafInstances)
    val rootTrainingNode: TrainingNode[AnyVal, Double] = if (split.isInstanceOf[NoSplit] || maxDepth == 0) {
      new RegressionTrainingLeaf(finalTraining, myLeafLearner, 0)
    } else {
      new RegressionTrainingNode(
        finalTraining,
        myLeafLearner,
        splitter,
        split,
        delta,
        numFeaturesActual,
        minLeafInstances = minLeafInstances,
        remainingDepth = maxDepth - 1,
        maxDepth
      )
    }

    /* Wrap them up in a regression tree */
    new RegressionTreeTrainingResult(rootTrainingNode, encoders)
  }

}

class RegressionTreeTrainingResult(
                                    rootTrainingNode: TrainingNode[AnyVal, Double],
                                    encoders: Seq[Option[CategoricalEncoder[Any]]]
                                  ) extends TrainingResult {
  lazy val model = new RegressionTree(rootTrainingNode.getNode(), encoders)
  lazy val importance = rootTrainingNode.getFeatureImportance()
  lazy val importanceNormalized = {
    if (Math.abs(importance.sum) > 0) {
      importance.map(_ / importance.sum)
    } else {
      importance.map(_ => 1.0 / importance.size)
    }
  }

  override def getModel(): RegressionTree = model

  /**
    * Return the pre-computed influences
    *
    * @return feature influences as an array of doubles
    */
  override def getFeatureImportance(): Option[Vector[Double]] = Some(importanceNormalized.toVector)
}

/**
  * Container holding a model node, encoders, and the feature influences
  *
  * @param root     of the tree
  * @param encoders for categorical variables
  */
class RegressionTree(
                      root: ModelNode[PredictionResult[Double]],
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
      inputs.map(inp => root.transform(CategoricalEncoder.encodeInput(inp, encoders)))
    )
  }
}

/**
  * Simple wrapper around a sequence of predictions
  *
  * @param predictions sequence of predictions
  */
class RegressionTreeResult(predictions: Seq[(PredictionResult[Double], TreeMeta)]) extends PredictionResult[Double] {
  /**
    * Get the predictions
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = predictions.map(_._1.getExpected().head)

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = {
    if (!predictions.head._1.getGradient().isDefined) {
      return None
    }
    Some(predictions.map(_._1.getGradient().get.head))
  }

  def getDepth(): Seq[Int] = {
    predictions.map(_._2.depth)
  }
}
