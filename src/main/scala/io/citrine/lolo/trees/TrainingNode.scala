package io.citrine.lolo.trees

import io.citrine.lolo.{Model, TrainingResult}

import scala.collection.mutable

/**
  * Trait to provide getNode interface for internal and leaf training nodes
  *
  * @tparam T type of the model output
  */
trait TrainingNode[+T] extends Serializable {

  // TODO (PLA-10433): make this a structured type
  def trainingData: Seq[(Vector[AnyVal], T, Double)]

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

  def modelNode: ModelNode[T] = ModelLeaf(model, depth, trainingData.map(_._3).sum)

  def model: Model[T] = trainingResult.getModel()
}
