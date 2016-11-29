package io.citrine.lolo.trees

/**
  * Class to provide getNode interface for internal and leaf training nodes
  * Created by maxhutch on 11/29/16.
  */
abstract class TrainingNode[T <: AnyVal](
                             trainingData: Seq[(Vector[T], Double)],
                             var impurity: Double = -1.0,
                             remainingDepth: Int = Int.MaxValue
                           ) {
  if (impurity < 0.0) {
    var sum = 0.0
    var sq = 0.0
    trainingData.foreach { case (x, y) =>
      sum = sum + y
      sq = sq + y * y
    }
    impurity = sq - sum * sum / trainingData.size
  }

  /**
    * Get the lightweight prediction node for the output tree
    * @return lightweight prediction node
    */
  def getNode(): ModelNode[T]

  def getImpurity(): Double = impurity

  def getFeatureImportance(): Array[Double]
}

/**
  * Trait to hold prediction interface
  */
abstract trait ModelNode[T <: AnyVal] {
  def predict(input: Vector[T]): Double
}

