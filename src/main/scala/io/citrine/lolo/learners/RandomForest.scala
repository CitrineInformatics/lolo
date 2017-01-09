package io.citrine.lolo.learners

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.trees.{ClassificationTreeLearner, RegressionTreeLearner}
import io.citrine.lolo.{Learner, TrainingResult}

/**
  * Created by maxhutch on 1/9/17.
  */
class RandomForest extends Learner {
  override var hypers: Map[String, Any] = Map(
    "numTrees" -> -1,
    "useJackknife" -> true,
    "biasLearner" -> None,
    "leafLearner" -> None,
    "subsetStrategy" -> "auto"
  )

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
        val numFeatures: Int = hypers("subsetStrategy") match {
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
        val DTLearner = new RegressionTreeLearner(
          leafLearner = hypers("leafLearner").asInstanceOf[Option[Learner]],
          numFeatures = numFeatures)
        val bagger = new Bagger(DTLearner,
          numBags = hypers("numTrees").asInstanceOf[Int],
          useJackknife = hypers("useJackknife").asInstanceOf[Boolean],
          biasLearner = hypers("biasLearner").asInstanceOf[Option[Learner]]
        )
        bagger.train(trainingData, weights)
      case x: Any =>
        val numFeatures: Int = hypers("subsetStrategy") match {
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
        val DTLearner = new ClassificationTreeLearner(numFeatures = numFeatures)
        val bagger = new Bagger(DTLearner,
          numBags = hypers("numTrees").asInstanceOf[Int]
        )
        bagger.train(trainingData, weights)
    }
  }
}
