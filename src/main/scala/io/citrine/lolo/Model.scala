package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
abstract class Model {

  /**
    * Apply the model to a seq of inputs
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  def transform(inputs: Seq[Vector[Any]]): PredictionResult

  /**
    * Get a measure of the importance of the model features
    * @return feature importances as an array of doubles
    */
  def getFeatureImportance(): Array[Double]
}
