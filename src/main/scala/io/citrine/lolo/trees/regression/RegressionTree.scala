package io.citrine.lolo.trees.regression

import breeze.linalg.DenseMatrix
import io.citrine.random.Random
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.splits.{RegressionSplitter, Splitter}
import io.citrine.lolo.trees.{ModelNode, TrainingNode, TreeMeta}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult, TrainingRow}

import scala.collection.mutable

/**
  * Learner for regression trees
  *
  * @param numFeatures to randomly select from at each split (default: all)
  * @param maxDepth    to grow the tree to
  * @param minLeafInstances minimum number of training instances per leaf
  * @param leafLearner learner to train the leaves with
  * @param splitter to determine the best split of the node data
  */
case class RegressionTreeLearner(
    numFeatures: Int = -1,
    maxDepth: Int = 30,
    minLeafInstances: Int = 1,
    leafLearner: Option[Learner[Double]] = None,
    splitter: Splitter[Double] = RegressionSplitter()
) extends Learner[Double] {

  /** Learner to use for training the leaves */
  @transient private lazy val myLeafLearner = leafLearner.getOrElse(GuessTheMeanLearner())

  /**
    * Train the tree by recursively partitioning (splitting) the training data on a single feature
    *
    * @param trainingData to train on
    * @param rng          random number generator for reproducibility
    * @return a RegressionTree
    */
  override def train(trainingData: Seq[TrainingRow[Double]], rng: Random): RegressionTreeTrainingResult = {
    require(trainingData.nonEmpty, s"The input training data was empty")

    /* Create encoders for any categorical features */
    val repInput = trainingData.head.inputs
    val encoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map {
      case (v, i) =>
        if (v.isInstanceOf[Double]) {
          None
        } else {
          Some(CategoricalEncoder.buildEncoder(trainingData.map(_.inputs(i))))
        }
    }

    // Encode the training data.
    val encodedTraining = trainingData.map { row =>
      val encodedInputs = CategoricalEncoder.encodeInput(row.inputs, encoders)
      row.withInputs(encodedInputs)
    }

    // Add the weights to the (features, label) tuples and remove any with zero weight.
    val finalTraining = encodedTraining.filter(_.weight > 0.0).toVector

    require(
      finalTraining.size >= 4,
      s"We need to have at least 4 rows with non-zero weights, only ${finalTraining.size} given"
    )

    // If the number of features isn't specified, use all of them.
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head.inputs.size
    }

    // Recursively build the tree via its nodes and wrap the top node in a RegressionTreeTrainingResult.
    val rootTrainingNode = RegressionTrainingNode.build(
      trainingData = finalTraining,
      leafLearner = myLeafLearner,
      splitter = splitter,
      numFeatures = numFeaturesActual,
      minLeafInstances = minLeafInstances,
      remainingDepth = maxDepth,
      maxDepth = maxDepth,
      rng = rng
    )
    RegressionTreeTrainingResult(rootTrainingNode, encoders)
  }

}

case class RegressionTreeTrainingResult(
    rootTrainingNode: TrainingNode[Double],
    encoders: Seq[Option[CategoricalEncoder[Any]]]
) extends TrainingResult[Double] {

  override lazy val model: RegressionTree = RegressionTree(rootTrainingNode.modelNode, encoders)

  lazy val nodeImportance: mutable.ArraySeq[Double] = rootTrainingNode.featureImportance

  override lazy val featureImportance: Option[Vector[Double]] = Some(
    if (Math.abs(nodeImportance.sum) > 0) {
      nodeImportance.map(_ / nodeImportance.sum).toVector
    } else {
      nodeImportance.map(_ => 1.0 / nodeImportance.size).toVector
    }
  )
}

/**
  * Container holding a model node, encoders, and the feature influences
  *
  * @param root     of the tree
  * @param encoders for categorical variables
  */
case class RegressionTree(
    root: ModelNode[Double],
    encoders: Seq[Option[CategoricalEncoder[Any]]]
) extends Model[Double] {

  /**
    * Apply the model by calling predict and wrapping the results
    *
    * @param inputs to apply the model to
    * @return a prediction result which includes only the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): RegressionTreePrediction = {
    RegressionTreePrediction(
      inputs.map(inp => root.transform(CategoricalEncoder.encodeInput(inp, encoders)))
    )
  }

  /**
    * Compute Shapley feature attributions for a given input
    *
    * @param input for which to compute feature attributions.
    * @param omitFeatures feature indices to omit in computing Shapley values
    * @return array of Shapley feature attributions, one per input feature, each a vector of
    *         One Vector[Double] per feature, each of length equal to the output dimension.
    *         The output dimension is 1 for single-task regression, or equal to the number of classification categories.
    */
  override def shapley(input: Vector[Any], omitFeatures: Set[Int] = Set()): Option[DenseMatrix[Double]] = {
    root.shapley(CategoricalEncoder.encodeInput(input, encoders), omitFeatures)
  }

}

/**
  * Simple wrapper around a sequence of predictions
  *
  * @param predictions sequence of predictions
  */
case class RegressionTreePrediction(predictions: Seq[(PredictionResult[Double], TreeMeta)])
    extends PredictionResult[Double] {

  /**
    * Get the predictions
    *
    * @return expected value of each prediction
    */
  override def expected: Seq[Double] = predictions.map(_._1.expected.head)

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def gradient: Option[Seq[Vector[Double]]] = {
    if (predictions.head._1.gradient.isEmpty) {
      return None
    }
    Some(predictions.map(_._1.gradient.get.head))
  }

  def depth: Seq[Int] = predictions.map(_._2.depth)
}
