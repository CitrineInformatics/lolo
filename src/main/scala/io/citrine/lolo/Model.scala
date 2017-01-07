package io.citrine.lolo

/**
  * Created by maxhutch on 11/14/16.
  */
@SerialVersionUID(1000L)
trait Model[+T <: PredictionResult[Any]] extends Serializable {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  def transform(inputs: Seq[Vector[Any]]): T
}
