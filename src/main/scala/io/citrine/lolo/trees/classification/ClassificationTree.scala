package io.citrine.lolo.trees.classification

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.trees.regression.RegressionTree
import io.citrine.lolo.trees.splits.{ClassificationSplitter, NoSplit, Split}
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingNode, TreeMeta}
import io.citrine.lolo.{Learner, Model, MultiResult, PredictionResult, TrainingResult}

import scala.collection.mutable

/**
  * Created by maxhutch on 12/2/16.
  *
  * @param numFeatures subset of features to select splits from
  */
class ClassificationTreeLearner(val numFeatures: Int = -1) extends Learner {

  override var hypers: Map[String, Any] = Map("maxDepth" -> 30)

  /**
    * Train classification tree
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a classification tree
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): ClassificationTrainingResult = {
    assert(trainingData.size > 4, s"We need to have at least 4 rows, only ${trainingData.size} given")
    val repInput = trainingData.head._1
    val maxDepth = hypers("maxDepth").asInstanceOf[Int]

    /* Create encoders for any categorical features */
    val inputEncoders: Seq[Option[CategoricalEncoder[Any]]] = repInput.zipWithIndex.map { case (v, i) =>
      if (v.isInstanceOf[Double]) {
        None
      } else {
        Some(CategoricalEncoder.buildEncoder(trainingData.map(_._1(i))))
      }
    }

    val outputEncoder = CategoricalEncoder.buildEncoder(trainingData.map(_._2))

    /* Encode the training data */
    val encodedTraining = trainingData.map(p =>
      (CategoricalEncoder.encodeInput(p._1, inputEncoders), outputEncoder.encode(p._2))
    )

    /* Add the weights to the (features, label) tuples and remove any with zero weight */
    val finalTraining = encodedTraining.zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0))).map { case ((f, l), w) =>
      (f, l, w)
    }.filter(_._3 > 0).toVector

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    /* The tree is built of training nodes */
    val (split, delta) = ClassificationSplitter.getBestSplit(finalTraining, numFeaturesActual)
    val rootTrainingNode = if (split.isInstanceOf[NoSplit]) {
      new ClassificationTrainingLeaf(finalTraining, 0)
    } else {
      new ClassificationTrainingNode(finalTraining, split, delta, numFeaturesActual, remainingDepth = maxDepth - 1, maxDepth)
    }

    /* Wrap them up in a regression tree */
    new ClassificationTrainingResult(rootTrainingNode, inputEncoders, outputEncoder, hypers)
  }
}

@SerialVersionUID(999L)
class ClassificationTrainingResult(
                                    rootTrainingNode: TrainingNode[AnyVal, Char],
                                    inputEncoders: Seq[Option[CategoricalEncoder[Any]]],
                                    outputEncoder: CategoricalEncoder[Any],
                                    hypers: Map[String, Any]
                                  ) extends TrainingResult {
  /* Grab a prediction node.  The partitioning happens here */
  lazy val model = new ClassificationTree(rootTrainingNode.getNode(), inputEncoders, outputEncoder)

  /* Grab the feature influences */
  lazy val importance = rootTrainingNode.getFeatureImportance()
  lazy val importanceNormalized = importance.map(_ / importance.sum)

  /**
    * Get the hyperparameters used to train this model
    *
    * @return hypers set for model
    */
  override def getHypers(): Map[String, Any] = hypers

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
@SerialVersionUID(999L)
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

class ClassificationTrainingNode(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                                  split: Split,
                                  deltaImpurity: Double,
                                  numFeatures: Int,
                                  remainingDepth: Int,
                                  maxDepth: Int
                                ) extends TrainingNode(trainingData, remainingDepth) {

  assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
  assert(!split.isInstanceOf[NoSplit], s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}")
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0 && leftTrain.exists(_._2 != leftTrain.head._2)) {
    lazy val (leftSplit, leftDelta) = ClassificationSplitter.getBestSplit(leftTrain, numFeatures)
    if (!leftSplit.isInstanceOf[NoSplit]) {
      new ClassificationTrainingNode(leftTrain, leftSplit, leftDelta, numFeatures, remainingDepth - 1, maxDepth)
    } else {
      new ClassificationTrainingLeaf(leftTrain, maxDepth - remainingDepth)
    }
  } else {
    new ClassificationTrainingLeaf(leftTrain, maxDepth - remainingDepth)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0 && rightTrain.exists(_._2 != rightTrain.head._2)) {
    lazy val (rightSplit, rightDelta) = ClassificationSplitter.getBestSplit(rightTrain, numFeatures)
    if (!rightSplit.isInstanceOf[NoSplit]) {
      new ClassificationTrainingNode(rightTrain, rightSplit, rightDelta, numFeatures, remainingDepth - 1, maxDepth)
    } else {
      new ClassificationTrainingLeaf(rightTrain, maxDepth - remainingDepth)
    }
  } else {
    new ClassificationTrainingLeaf(rightTrain, maxDepth - remainingDepth)
  }

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Char]] = new InternalModelNode(
    split, leftChild.getNode(), rightChild.getNode()
  )

  override def getFeatureImportance(): mutable.ArraySeq[Double] = {
    val improvement = deltaImpurity
    var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex) = ans(split.getIndex) + improvement
    ans
  }
}

class ClassificationTrainingLeaf(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                                  depth: Int
                                ) extends TrainingNode(trainingData, 0) {

  lazy val mode: Char = trainingData.map(_._2).groupBy(identity).mapValues(_.size).maxBy(_._2)._1

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Char]] = new ClassificationLeaf(mode, depth)

  override def getFeatureImportance(): mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
}

/**
  * The leaves just predict the most common value
  *
  * @param mode most common value
  */
class ClassificationLeaf(mode: Char, depth: Int) extends ModelNode[PredictionResult[Char]] {
  override def transform(input: Vector[AnyVal]): (PredictionResult[Char], TreeMeta) = (MultiResult(Seq(mode)), TreeMeta(depth))
}