package io.citrine.lolo.trees.classification

import io.citrine.lolo.{Learner, Model, PredictionResult}
import io.citrine.lolo.trees.{ModelLeaf, ModelNode, TrainingNode}

import scala.collection.mutable

/**
  * Created by maxhutch on 1/12/17.
  */
class ClassificationTrainingLeaf(
                                  trainingData: Seq[(Vector[AnyVal], Char, Double)],
                                  leafLearner: Learner,
                                  depth: Int
                                ) extends TrainingNode(trainingData, 0) {

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  override def getNode(): ModelNode[PredictionResult[Char]] = {
    new ModelLeaf(
      leafLearner.train(trainingData)
        .getModel()
        .asInstanceOf[Model[PredictionResult[Char]]],
      depth
    )
  }

  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
}
