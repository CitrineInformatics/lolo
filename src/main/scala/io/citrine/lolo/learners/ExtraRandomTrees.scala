package io.citrine.lolo.learners

import io.citrine.random.Random
import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.transformers.FeatureRotator
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{ExtraRandomClassificationSplitter, ExtraRandomRegressionSplitter}
import io.citrine.lolo.{Learner, TrainingResult}

/**
  * Extremely randomized tree ensemble
  *
  * This is based on Geurts, P., Ernst, D. & Wehenkel, L. Extremely randomized trees. Mach Learn 63, 3–42 (2006).
  * https://doi.org/10.1007/s10994-006-6226-1.
  *
  * @param numTrees       number of trees to use (-1 => number of training instances)
  * @param useJackknife   whether to use jackknife based variance estimates
  * @param biasLearner    learner to model bias (absolute residual)
  * @param leafLearner    learner to use at the leaves of the trees
  * @param subsetStrategy for random feature selection at each split
  *                       (auto => all features for regression)
  * @param minLeafInstances minimum number of instances per leave in each tree
  * @param maxDepth       maximum depth of each tree in the forest (default: unlimited)
  * @param uncertaintyCalibration whether to empirically recalibrate the predicted uncertainties (default: false)
  * @param disableBootstrap whether to disable bootstrap (default: true)
  * @param randomlyRotateFeatures whether to randomly rotate real features for each tree in the forest (default: false)
  */
case class ExtraRandomTrees(
    numTrees: Int = -1,
    useJackknife: Boolean = false,
    biasLearner: Option[Learner[Double]] = None,
    leafLearner: Option[Learner[Any]] = None,
    subsetStrategy: Any = "auto",
    minLeafInstances: Int = 1,
    maxDepth: Int = Integer.MAX_VALUE,
    uncertaintyCalibration: Boolean = false,
    disableBootstrap: Boolean = true,
    randomlyRotateFeatures: Boolean = false
) extends Learner[Any] {

  /**
    * Train an extremely randomized tree ensemble model
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
  ): TrainingResult[Any] = {
    val repOutput = trainingData.head._2
    val isRegression = repOutput.isInstanceOf[Double]
    val numFeatures: Int = RandomForest.getNumFeatures(subsetStrategy, trainingData.head._1.size, isRegression)

    repOutput match {
      case _: Double =>
        val DTLearner = RegressionTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = ExtraRandomRegressionSplitter()
        )
        val bagger = Bagger(
          if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration,
          disableBootstrap = disableBootstrap
        )
        bagger.train(trainingData, weights, rng)
      case _: Any =>
        val DTLearner = ClassificationTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures,
          minLeafInstances = minLeafInstances,
          maxDepth = maxDepth,
          splitter = ExtraRandomClassificationSplitter()
        )
        val bagger = Bagger(
          if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner,
          uncertaintyCalibration = uncertaintyCalibration,
          disableBootstrap = disableBootstrap
        )
        bagger.train(trainingData, weights, rng)
    }
  }
}
