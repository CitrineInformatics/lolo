package io.citrine.lolo.trees

import io.citrine.lolo.{Model, TrainingResult, TrainingRow}

import scala.collection.mutable

/**
  * Trait to provide getNode interface for internal and leaf training nodes
  *
  * @tparam T type of the model output
  */
trait TrainingNode[+T] extends Serializable {

  def trainingData: Seq[TrainingRow[T]]

  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  def modelNode: ModelNode[T]

  /**
    * Get the feature importance of the subtree below this node
    *
    * @return feature importance as a vector
    */
  def featureImportance: mutable.ArraySeq[Double]
}

/** A leaf defined by a training result. */
trait TrainingLeaf[+T] extends TrainingNode[T] {

  def depth: Int

  def trainingResult: TrainingResult[T]

  def modelNode: ModelNode[T] = ModelLeaf(model, depth, trainingData.map(_.weight).sum)

  def model: Model[T] = trainingResult.model
}
