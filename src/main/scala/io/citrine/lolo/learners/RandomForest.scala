package io.citrine.lolo.learners

import io.citrine.random.Random
import io.citrine.lolo.bags.{Bagger, MultiTaskBagger}
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.transformers.{FeatureRotator, MultiTaskFeatureRotator, MultiTaskStandardizer}
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{ClassificationSplitter, MultiTaskSplitter, RegressionSplitter}
import io.citrine.lolo.{Learner, TrainingResult}

/**
  * Standard random forest as a wrapper around bagged decision trees
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
  * @param randomizePivotLocation whether to generate splits randomly between the data points (default: false)
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
    uncertaintyCalibration: Boolean = true,
    randomizePivotLocation: Boolean = false,
    randomlyRotateFeatures: Boolean = false
) extends Learner {

  /**
    * Train a random forest model
    *
    * If the training label is a Double then this is a regression forest.
    * If the training label is another primtiive then this is a classification forest.
    * If the training label is a sequence then this is a multitask forest.
    * Options like the number of trees are set via the constructor.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return training result containing a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]],
      rng: Random
  ): TrainingResult = {
    val repOutput = trainingData.head._2
    val isRegression = repOutput.isInstanceOf[Double] ||
      (repOutput.isInstanceOf[Seq[Any]] && repOutput.asInstanceOf[Seq[Any]].exists(_.isInstanceOf[Double]))
    val numFeatures: Int = RandomForest.getNumFeatures(subsetStrategy, trainingData.head._1.size, isRegression)

    repOutput match {
      case _: Double =>
        val DTLearner = RegressionTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = RegressionSplitter(randomizePivotLocation)
        )

        val bagger = Bagger(
          if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration
        )
        bagger.train(trainingData, weights, rng)
      case _: Seq[Any] =>
        if (leafLearner.isDefined && !leafLearner.get.isInstanceOf[GuessTheMeanLearner]) {
          throw new IllegalArgumentException(
            "Multitask random forest does not support leaf learners other than guess-the-mean"
          )
        }
        val DTLearner = new MultiTaskStandardizer(
          MultiTaskTreeLearner(
            numFeatures = numFeatures,
            maxDepth = maxDepth,
            minLeafInstances = minLeafInstances,
            splitter = MultiTaskSplitter(randomizePivotLocation)
          )
        )
        val bagger = MultiTaskBagger(
          if (randomlyRotateFeatures) MultiTaskFeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration
        )
        bagger.train(trainingData.asInstanceOf[Seq[(Vector[Any], Vector[Any])]], weights, rng)
      case _: Any =>
        val DTLearner = ClassificationTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = ClassificationSplitter(randomizePivotLocation)
        )
        val bagger = Bagger(
          if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees
        )
        bagger.train(trainingData, weights, rng)
    }
  }
}

object RandomForest {

  def getNumFeatures(subsetStrategy: Any, numTrainingFeatures: Int, isRegression: Boolean): Int = {
    subsetStrategy match {
      case x: String =>
        x match {
          case "auto" =>
            if (isRegression) numTrainingFeatures else Math.ceil(Math.sqrt(numTrainingFeatures)).toInt
          case "sqrt" => Math.ceil(Math.sqrt(numTrainingFeatures)).toInt
          case "log2" => Math.ceil(Math.log(numTrainingFeatures) / Math.log(2)).toInt
          case x: String =>
            println(s"Unrecognized subsetStrategy $x; using auto")
            getNumFeatures("auto", numTrainingFeatures, isRegression)
        }
      case x: Int =>
        x
      case x: Double =>
        (numTrainingFeatures * x).toInt
    }
  }

}
