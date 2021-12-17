package io.citrine.lolo.learners

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.bags.{Bagger, MultiTaskBagger}
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.transformers.{FeatureRotator, MultiTaskFeatureRotator, MultiTaskStandardizer}
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{ClassificationSplitter, RegressionSplitter}
import io.citrine.lolo.{Learner, TrainingResult}
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

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
  * @param randomizePivotLocation whether to generate splits randomly between the data points (default: false)
  * @param randomlyRotateFeatures whether to randomly rotate real features for each tree in the forest (default: false)
  * @param rng            random number generator to use for stochastic functionality
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
                         randomlyRotateFeatures: Boolean = false,
                         rng: Random = Random
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
    val breezeRandBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(rng.nextLong())))
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
          splitter = RegressionSplitter(randomizePivotLocation, rng = new Random(rng.nextLong())),
          rng = new Random(rng.nextLong())
        )

        val bagger = Bagger(
          if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration,
          randBasis = breezeRandBasis
        )
        bagger.train(trainingData, weights)
      case _: Seq[Any] =>
        if (leafLearner.isDefined && !leafLearner.get.isInstanceOf[GuessTheMeanLearner]) {
          throw new IllegalArgumentException("Multitask random forest does not support leaf learners other than guess-the-mean")
        }
        val DTLearner = new MultiTaskStandardizer(MultiTaskTreeLearner(
          numFeatures = numFeatures,
          maxDepth = maxDepth,
          minLeafInstances = minLeafInstances,
          randomizePivotLocation = randomizePivotLocation,
          rng = new Random(rng.nextLong())
        ))
        val bagger = MultiTaskBagger(
           if (randomlyRotateFeatures) MultiTaskFeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration,
          randBasis = breezeRandBasis
        )
        bagger.train(trainingData.asInstanceOf[Seq[(Vector[Any], Vector[Any])]], weights)
      case _: Any =>
        val DTLearner = ClassificationTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = ClassificationSplitter(randomizePivotLocation, rng = new Random(rng.nextLong())),
          rng = new Random(rng.nextLong())
        )
        val bagger = Bagger(
          if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          randBasis = breezeRandBasis
        )
        bagger.train(trainingData, weights)
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