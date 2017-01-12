package io.citrine.lolo.trees.classification

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.trees.{ModelNode, TrainingNode}

import scala.collection.mutable

/**
  * Created by maxhutch on 1/12/17.
  */
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

  override def getFeatureImportance(): scala.collection.mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
}
