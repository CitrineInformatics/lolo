package io.citrine.lolo.api

trait TrainingResult[+T] extends Serializable {

  /**
    * Get the model contained in the training result
    *
    * @return the model
    */
  def model: Model[T]

  /**
    * Get a measure of the importance of the model features
    *
    * @return feature influences as an array of doubles
    */
  def featureImportance: Option[Vector[Double]] = None

  /**
    * Get a measure of the loss of the model, e.g. RMS OOB error
    *
    * @return
    */
  def loss: Option[Double] = None

  /**
    * Get the predicted vs actual values, e.g. from OOB
    *
    * @return seq of (feature vector, predicted value, and actual value)
    */
  def predictedVsActual: Option[Seq[(Vector[Any], T, T)]] = None
}

trait MultiTaskTrainingResult extends TrainingResult[Vector[Any]] {

  def models: Seq[Model[Any]]

  override def model: MultiTaskModel

  override def predictedVsActual: Option[Seq[(Vector[Any], Vector[Option[Any]], Vector[Option[Any]])]] = None
}
