package io.citrine.lolo.learners

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
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
  *                       (auto => 1/3 for regression, sqrt for classification)
  */
case class RandomForest(
                         numTrees: Int = -1,
                         useJackknife: Boolean = true,
                         biasLearner: Option[Learner] = None,
                         leafLearner: Option[Learner] = None,
                         subsetStrategy: Any = "auto"
                       ) extends Learner {

  override def getHypers(): Map[String, Any] = {
    Map(
      "numTrees" -> numTrees,
      "useJackknife" -> useJackknife,
      "biasLearner" -> biasLearner,
      "leafLearner" -> leafLearner,
      "subsetStrategy" -> subsetStrategy
    )
  }

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
      case x: Double =>
        val numFeatures: Int = subsetStrategy match {
          case x: String =>
            x match {
              case "auto" => (trainingData.head._1.size / 3.0).toInt
              case "sqrt" => Math.sqrt(trainingData.head._1.size).toInt
              case x: String =>
                println(s"Unrecognized subsetStrategy ${x}; using auto")
                (trainingData.head._1.size / 3.0).toInt
            }
          case x: Int =>
            x
          case x: Double =>
            (trainingData.head._1.size * x).toInt
        }
        val DTLearner = RegressionTreeLearner(
          leafLearner = leafLearner,
          numFeatures = numFeatures)
        val bagger = Bagger(DTLearner,
          numBags = numTrees,
          useJackknife = useJackknife,
          biasLearner = biasLearner
        )
        bagger.train(trainingData, weights)
      case x: Any =>
        val numFeatures: Int = subsetStrategy match {
          case x: String =>
            x match {
              case "auto" => Math.sqrt(trainingData.head._1.size).toInt
              case "sqrt" => Math.sqrt(trainingData.head._1.size).toInt
              case x: String =>
                println(s"Unrecognized subsetStrategy ${x}; using auto")
                Math.sqrt(trainingData.head._1.size).toInt
            }
          case x: Int =>
            x
          case x: Double =>
            (trainingData.head._1.size * x).toInt
        }
        val DTLearner = ClassificationTreeLearner(numFeatures = numFeatures)
        val bagger = Bagger(DTLearner,
          numBags = numTrees
        )
        bagger.train(trainingData, weights)
    }
  }
}
