package io.citrine.lolo.trees.classification

import io.citrine.random.Random
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.splits.{ClassificationSplitter, NoSplit, Splitter}
import io.citrine.lolo.trees.{ModelNode, TrainingLeaf, TrainingNode, TreeMeta}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

/**
  * @param numFeatures subset of features to select splits from
  * @param maxDepth maximum depth of tree
  * @param minLeafInstances minimum training instances per node
  * @param leafLearner to train on leaves
  * @param splitter used to select splits
  */
case class ClassificationTreeLearner(
    numFeatures: Int = -1,
    maxDepth: Int = 30,
    minLeafInstances: Int = 1,
    leafLearner: Option[Learner] = None,
    splitter: Splitter[Char] = ClassificationSplitter()
) extends Learner {

  @transient private lazy val myLeafLearner: Learner = leafLearner.getOrElse(GuessTheMeanLearner())

  /**
    * Train classification tree
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a classification tree
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): ClassificationTrainingResult = {
    assert(trainingData.size > 4, s"We need to have at least 4 rows, only ${trainingData.size} given")
    val repInput = trainingData.head._1

    /* Create encoders for any categorical features */
    val inputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map {
      case (v, i) =>
        if (v.isInstanceOf[Double]) {
          None
        } else {
          Some(CategoricalEncoder.buildEncoder(trainingData.map(_._1(i))))
        }
    }

    val outputEncoder = CategoricalEncoder.buildEncoder(trainingData.map(_._2))

    /* Encode the training data */
    val encodedTraining =
      trainingData.map(p => (CategoricalEncoder.encodeInput(p._1, inputEncoders), outputEncoder.encode(p._2)))

    /* Add the weights to the (features, label) tuples and remove any with zero weight */
    val finalTraining = encodedTraining
      .zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0)))
      .map {
        case ((f, l), w) =>
          (f, l, w)
      }
      .filter(_._3 > 0)
      .toVector

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    // Recursively build the tree via it snodes and wrap the top node in a ClassificationTreeTrainingResult
    val rootTrainingNode = ClassificationTrainingNode.build(
      trainingData = finalTraining,
      leafLearner = myLeafLearner,
      splitter = splitter,
      numFeatures = numFeaturesActual,
      minLeafInstances = minLeafInstances,
      remainingDepth = maxDepth,
      maxDepth = maxDepth,
      numClasses = trainingData.map(_._2).distinct.length,
      rng = rng
    )
    new ClassificationTrainingResult(rootTrainingNode, inputEncoders, outputEncoder)
  }
}

class ClassificationTrainingResult(
    rootTrainingNode: TrainingNode[Char],
    inputEncoders: Seq[Option[CategoricalEncoder[Any]]],
    outputEncoder: CategoricalEncoder[Any]
) extends TrainingResult {
  /* Grab a prediction node.  The partitioning happens here */
  lazy val model = new ClassificationTree(rootTrainingNode.getModelNode(), inputEncoders, outputEncoder)

  /* Grab the feature influences */
  lazy val importance = rootTrainingNode.getFeatureImportance()
  private lazy val importanceNormalized = {
    if (Math.abs(importance.sum) > 0) {
      importance.map(_ / importance.sum)
    } else {
      importance.map(_ => 1.0 / importance.size)
    }
  }

  override def getModel(): ClassificationTree = model

  /**
    * Get a measure of the importance of the model features
    *
    * @return feature influences as an array of doubles
    */
  override def getFeatureImportance(): Option[Vector[Double]] = Some(importanceNormalized.toVector)
}

/**
  * Classification tree
  */
class ClassificationTree(
    rootModelNode: ModelNode[PredictionResult[Char]],
    inputEncoders: Seq[Option[CategoricalEncoder[Any]]],
    outputEncoder: CategoricalEncoder[Any]
) extends Model[ClassificationResult] {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): ClassificationResult = {
    new ClassificationResult(
      inputs.map(inp => rootModelNode.transform(CategoricalEncoder.encodeInput(inp, inputEncoders))),
      outputEncoder
    )
  }
}

/**
  * Classification result
  */
class ClassificationResult(
    predictions: Seq[(PredictionResult[Char], TreeMeta)],
    outputEncoder: CategoricalEncoder[Any]
) extends PredictionResult[Any] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = predictions.map(p => outputEncoder.decode(p._1.getExpected().head))

  def getDepth(): Seq[Int] = {
    predictions.map(_._2.depth)
  }
}
