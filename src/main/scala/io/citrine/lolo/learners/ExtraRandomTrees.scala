package io.citrine.lolo.learners

import io.citrine.lolo.api.{Learner, TrainingResult, TrainingRow}
import io.citrine.random.Random
import io.citrine.lolo.bags.{ClassificationBagger, RegressionBagger}
import io.citrine.lolo.transformers.rotator.FeatureRotator
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{ExtraRandomClassificationSplitter, ExtraRandomRegressionSplitter}

/**
  * Extremely randomized tree ensemble for regression.
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
case class ExtraRandomTreesRegressor(
    numTrees: Int = -1,
    useJackknife: Boolean = false,
    biasLearner: Option[Learner[Double]] = None,
    leafLearner: Option[Learner[Double]] = None,
    subsetStrategy: Any = "auto",
    minLeafInstances: Int = 1,
    maxDepth: Int = Integer.MAX_VALUE,
    uncertaintyCalibration: Boolean = false,
    disableBootstrap: Boolean = true,
    randomlyRotateFeatures: Boolean = false
) extends Learner[Double] {

  override def train(trainingData: Seq[TrainingRow[Double]], rng: Random): TrainingResult[Double] = {
    val numFeatures = RandomForest.getNumFeatures(subsetStrategy, trainingData.head.inputs.length, isRegression = true)

    val DTLearner = RegressionTreeLearner(
      leafLearner = leafLearner,
      numFeatures = numFeatures,
      minLeafInstances = minLeafInstances,
      maxDepth = maxDepth,
      splitter = ExtraRandomRegressionSplitter()
    )
    val bagger = RegressionBagger(
      if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
      numBags = numTrees,
      useJackknife = useJackknife,
      biasLearner = biasLearner,
      uncertaintyCalibration = uncertaintyCalibration,
      disableBootstrap = disableBootstrap
    )
    bagger.train(trainingData, rng)
  }
}

/**
  * Extremely randomized tree ensemble for classification.
  *
  * This is based on Geurts, P., Ernst, D. & Wehenkel, L. Extremely randomized trees. Mach Learn 63, 3–42 (2006).
  * https://doi.org/10.1007/s10994-006-6226-1.
  *
  * @param numTrees         number of trees to use (-1 => number of training instances)
  * @param useJackknife     whether to use jackknife based variance estimates
  * @param leafLearner      learner to use at the leaves of the trees
  * @param subsetStrategy   for random feature selection at each split
  *                         (auto => all features for regression)
  * @param minLeafInstances minimum number of instances per leave in each tree
  * @param maxDepth         maximum depth of each tree in the forest (default: unlimited)
  * @param disableBootstrap whether to disable bootstrap (default: true)
  * @param randomlyRotateFeatures whether to randomly rotate real features for each tree in the forest (default: false)
  */
case class ExtraRandomTreesClassifier(
    numTrees: Int = -1,
    useJackknife: Boolean = false,
    leafLearner: Option[Learner[Char]] = None,
    subsetStrategy: Any = "auto",
    minLeafInstances: Int = 1,
    maxDepth: Int = Integer.MAX_VALUE,
    disableBootstrap: Boolean = true,
    randomlyRotateFeatures: Boolean = false
) extends Learner[Any] {

  override def train(trainingData: Seq[TrainingRow[Any]], rng: Random): TrainingResult[Any] = {
    val numFeatures = RandomForest.getNumFeatures(subsetStrategy, trainingData.head.inputs.length, isRegression = false)

    val DTLearner = ClassificationTreeLearner(
      leafLearner = leafLearner,
      numFeatures = numFeatures,
      minLeafInstances = minLeafInstances,
      maxDepth = maxDepth,
      splitter = ExtraRandomClassificationSplitter()
    )
    val bagger = ClassificationBagger(
      if (randomlyRotateFeatures) FeatureRotator(DTLearner) else DTLearner,
      numBags = numTrees,
      useJackknife = useJackknife,
      disableBootstrap = disableBootstrap
    )
    bagger.train(trainingData, rng)
  }
}
