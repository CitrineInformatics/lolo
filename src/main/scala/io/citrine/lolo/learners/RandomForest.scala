package io.citrine.lolo.learners

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.transformers.FeatureRotator
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{ClassificationSplitter, RegressionSplitter}
import io.citrine.lolo.{Learner, TrainingResult}

/**
  * Standard random forest as a wrapper around bagged decision trees
  * Created by maxhutch on 1/9/17.
  *
  * @param numTrees       number of trees to use (-1 => number of training instances)
  * @param useJackknife   whether to use jackknife based variance estimates
  * @param biasLearner    learner to model bias (absolute residual)
  * @param leafLearner    learner to use at the leaves of the trees
  * @param subsetStrategy for random feature selection at each split
  *                       (auto => all fetures for regression, sqrt for classification)
  * @param minLeafInstances minimum number of instances per leave in each tree
  * @param maxDepth       maximum depth of each tree in the forest (default: unlimited)
  * @param uncertaintyCalibration whether to empirically recalibrate the predicted uncertainties (default: false)
  * @param randomizePivotLocation whether generate splits randomly between the data points (default: false)
  * @param randomlyRotateFeatures whether to randomly rotate real features for each tree in the forest (default: false)
  */
case class RandomForest(
                         numTrees: Int = -1,
                         useJackknife: Boolean = true,
                         biasLearner: Option[Learner] = None,
                         leafLearner: Option[Learner] = None,
                         subsetStrategy: Any = "auto",
                         minLeafInstances: Int = 1,
                         maxDepth: Int = Integer.MAX_VALUE,
                         uncertaintyCalibration: Boolean = false,
                         randomizePivotLocation: Boolean = false,
                         randomlyRotateFeatures: Boolean = false
                       ) extends Learner {
  /**
    * Train a random forest model
    *
    * If the training labels are Doubles, this is a regression forest; otherwise, a classification forest.
    * Options like the number of trees are set via setHyper
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return training result containing a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): TrainingResult = {
    trainingData.head._2 match {
      case _: Double =>
        val numFeatures: Int = subsetStrategy match {
          case x: String =>
            x match {
              case "auto" => trainingData.head._1.size
              case "sqrt" => Math.ceil(Math.sqrt(trainingData.head._1.size)).toInt
              case "log2" => Math.ceil(Math.log(trainingData.head._1.size) / Math.log(2)).toInt
              case x: String =>
                println(s"Unrecognized subsetStrategy ${x}; using auto")
                trainingData.head._1.size
            }
          case x: Int =>
            x
          case x: Double =>
            (trainingData.head._1.size * x).toInt
        }
        val DTLearner = RegressionTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = RegressionSplitter(randomizePivotLocation)
        )
        val bagger = Bagger(if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration
        )
        bagger.train(trainingData, weights)
      case _: Any =>
        val numFeatures: Int = subsetStrategy match {
          case x: String =>
            x match {
              case "auto" => Math.ceil(Math.sqrt(trainingData.head._1.size)).toInt
              case "sqrt" => Math.ceil(Math.sqrt(trainingData.head._1.size)).toInt
              case "log2" => Math.ceil(Math.log(trainingData.head._1.size) / Math.log(2)).toInt
              case x: String =>
                println(s"Unrecognized subsetStrategy ${x}; using auto")
                Math.sqrt(trainingData.head._1.size).toInt
            }
          case x: Int =>
            x
          case x: Double =>
            (trainingData.head._1.size * x).toInt
        }
        val DTLearner = ClassificationTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = ClassificationSplitter(randomizePivotLocation)
        )
        val bagger = Bagger(if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees
        )
        bagger.train(trainingData, weights)
    }
  }
}
