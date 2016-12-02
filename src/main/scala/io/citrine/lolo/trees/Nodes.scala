package io.citrine.lolo.trees

/**
  * Class to provide getNode interface for internal and leaf training nodes
  * Created by maxhutch on 11/29/16.
  */
abstract class TrainingNode[T <: AnyVal](
                                          trainingData: Seq[(Vector[T], Double, Double)],
                                          remainingDepth: Int = Int.MaxValue
                                        ) {
  /**
    * Get the lightweight prediction node for the output tree
    *
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[T, Double]

  def getFeatureImportance(): Array[Double]
}

/**
  * Trait to hold prediction interface
  */
abstract trait ModelNode[T <: AnyVal, S] {
  def predict(input: Vector[T]): S
}

