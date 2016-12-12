package io.citrine.lolo.trees

import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.lolo.results.{MultiResult, PredictionResult, TrainingResult, hasFeatureImportance}
import io.citrine.lolo.trees.splits.{ClassificationSplitter, NoSplit, Split}
import io.citrine.lolo.{Learner, Model}

/**
  * Created by maxhutch on 12/2/16.
  *
  * @param numFeatures subset of features to select splits from
  */
class ClassificationTreeLearner(val numFeatures: Int = -1) extends Learner {

  override var hypers: Map[String, Any] = Map()

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
      (RegressionTree.encodeInput(p._1, inputEncoders), outputEncoder.encode(p._2))
    )

    /* Add the weights to the (features, label) tuples and remove any with zero weight */
    val finalTraining = encodedTraining.zip(weights.getOrElse(Seq.fill(trainingData.size)(1.0))).map { case ((f, l), w) =>
      (f, l, w)
    }.filter(_._3 > 0)

    /* If the number of features isn't specified, use all of them */
    val numFeaturesActual = if (numFeatures > 0) {
      numFeatures
    } else {
      finalTraining.head._1.size
    }

    /* The tree is built of training nodes */
    val (split, delta) = ClassificationSplitter.getBestSplit(finalTraining, numFeaturesActual)
    // assert(!split.isInstanceOf[NoSplit], s"Couldn't make a single split of ${numFeaturesActual} on ${finalTraining.size} rows: ${finalTraining}")
    val rootTrainingNode = if (split.isInstanceOf[NoSplit]) {
      new ClassificationTrainingLeaf(finalTraining)
    } else {
      new ClassificationTrainingNode(finalTraining, split, delta, numFeaturesActual, remainingDepth = 30)
    }

    /* Wrap them up in a regression tree */
    new ClassificationTrainingResult(rootTrainingNode, inputEncoders, outputEncoder)
  }
}

class ClassificationTrainingResult(
                                    rootTrainingNode: TrainingNode[AnyVal, Char],
                                    inputEncoders: Seq[Option[CategoricalEncoder[Any]]],
                                    outputEncoder: CategoricalEncoder[Any]
                                  ) extends TrainingResult with hasFeatureImportance {
  /* Grab a prediction node.  The partitioning happens here */
  lazy val model = new ClassificationTree(rootTrainingNode.getNode(), inputEncoders, outputEncoder)

  /* Grab the feature importances */
  lazy val importance = rootTrainingNode.getFeatureImportance()
  lazy val importanceNormalized = importance.map(_ / importance.sum)

  override def getModel(): ClassificationTree = model

  /**
    * Get a measure of the importance of the model features
    *
    * @return feature importances as an array of doubles
    */
  override def getFeatureImportance(): Array[Double] = importanceNormalized
}

/**
  * Classification tree
  */
class ClassificationTree(
                          rootModelNode: Model[PredictionResult[Char]],
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
      inputs.map(inp => outputEncoder.decode(rootModelNode.transform(Seq(RegressionTree.encodeInput(inp, inputEncoders))).getExpected().head))
    )
  }
}

/**
  * Classification result
  */
class ClassificationResult(predictions: Seq[Any]) extends PredictionResult[Any] {

  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Any] = predictions
}

class ClassificationTrainingNode(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                                  split: Split,
                                  deltaImpurity: Double,
                                  numFeatures: Int,
                                  remainingDepth: Int = Int.MaxValue
                                ) extends TrainingNode(trainingData, remainingDepth) {

  assert(trainingData.size > 1, "If we are going to split, we need at least 2 training rows")
  assert(!split.isInstanceOf[NoSplit], s"Empty split split for training data: \n${trainingData.map(_.toString() + "\n")}")

  lazy val (leftTrain, rightTrain) = trainingData.partition(r => split.turnLeft(r._1))
  assert(leftTrain.size > 0 && rightTrain.size > 0, s"Split ${split} resulted in zero size: ${trainingData.map(_._1(split.getIndex()))}")
  lazy val leftChild = if (leftTrain.size > 1 && remainingDepth > 0 && leftTrain.exists(_._2 != leftTrain.head._2)) {
    lazy val (leftSplit, leftDelta) = ClassificationSplitter.getBestSplit(leftTrain, numFeatures)
    if (!leftSplit.isInstanceOf[NoSplit]) {
      new ClassificationTrainingNode(leftTrain, leftSplit, leftDelta, numFeatures, remainingDepth - 1)
    } else {
      new ClassificationTrainingLeaf(leftTrain)
    }
  } else {
    new ClassificationTrainingLeaf(leftTrain)
  }
  lazy val rightChild = if (rightTrain.size > 1 && remainingDepth > 0 && rightTrain.exists(_._2 != rightTrain.head._2)) {
    lazy val (rightSplit, rightDelta) = ClassificationSplitter.getBestSplit(rightTrain, numFeatures)
    if (!rightSplit.isInstanceOf[NoSplit]) {
      new ClassificationTrainingNode(rightTrain, rightSplit, rightDelta, numFeatures, remainingDepth - 1)
    } else {
      new ClassificationTrainingLeaf(rightTrain)
    }
  } else {
    new ClassificationTrainingLeaf(rightTrain)
  }

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): Model[PredictionResult[Char]] = new InternalModelNode(
    split, leftChild.getNode(), rightChild.getNode()
  )

  override def getFeatureImportance(): Array[Double] = {
    val improvement = deltaImpurity
    var ans = leftChild.getFeatureImportance().zip(rightChild.getFeatureImportance()).map(p => p._1 + p._2)
    ans(split.getIndex) = ans(split.getIndex) + improvement
    ans
  }
}

class ClassificationTrainingLeaf(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)]
                                ) extends TrainingNode(trainingData, 0) {

  lazy val mode: Char = trainingData.map(_._2).groupBy(identity).mapValues(_.size).maxBy(_._2)._1

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): Model[PredictionResult[Char]] = new ClassificationLeaf(mode)

  override def getFeatureImportance(): Array[Double] = Array.fill(trainingData.head._1.size)(0.0)
}

/**
  * The leaves just predict the most common value
  *
  * @param mode most common value
  */
class ClassificationLeaf(mode: Char) extends Model[PredictionResult[Char]] {
  override def transform(inputs: Seq[Vector[Any]]): PredictionResult[Char] = MultiResult(Seq.fill(inputs.size)(mode))
}