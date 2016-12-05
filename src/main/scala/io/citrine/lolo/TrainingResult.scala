package io.citrine.lolo

/**
  * Created by maxhutch on 12/4/16.
  */
abstract trait TrainingResult extends Serializable {

  def getModel(): Model
}

abstract trait hasFeatureImportance {
    /**
    * Get a measure of the importance of the model features
    * @return feature importances as an array of doubles
    */
  def getFeatureImportance(): Array[Double]
}

abstract trait hasLoss {
  def getLoss(): Double
}

abstract trait hasPredictedVsActual {
  def getPredictedVsActual(): Seq[(Vector[Any], Any, Any)]
}

