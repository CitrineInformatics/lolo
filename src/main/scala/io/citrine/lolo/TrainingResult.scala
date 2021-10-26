package io.citrine.lolo

/**
  * Created by maxhutch on 12/4/16.
  */
trait TrainingResult extends Serializable {

  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  def getModel(): Model[PredictionResult[Any]]

  /**
    * Get a measure of the importance of the model features
    *
    * @return feature influences as an array of doubles
    */
  def getFeatureImportance(): Option[Vector[Double]] = None

  /**
    * Get a measure of the loss of the model, e.g. RMS OOB error
    *
    * @return
    */
  def getLoss(): Option[Double] = None

  /**
    * Get the predicted vs actual values, e.g. from OOB
    *
    * @return seq of (feature vector, predicted value, and actual value)
    */
  def getPredictedVsActual(): Option[Seq[(Vector[Any], Any, Any)]] = None
}

trait MultiTaskTrainingResult extends TrainingResult {
  override def getModel(): MultiTaskModel

  def getModels(): Seq[Model[PredictionResult[Any]]]

  override def getPredictedVsActual(): Option[Seq[(Vector[Any], Seq[Any], Seq[Any])]] = None
}