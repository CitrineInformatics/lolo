package io.citrine.lolo

/** Created by maxhutch on 11/14/16.
  */
trait Learner extends Serializable {

  /** Train a model
    *
    * @param trainingData
    *   to train on
    * @param weights
    *   for the training rows, if applicable
    * @return
    *   training result containing a model
    */
  def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): TrainingResult

  /** Train a model with weights
    *
    * @param trainingData
    *   with weights in the form (features, label, weight)
    * @return
    *   training result containing a model
    */
  def train(trainingData: Seq[(Vector[Any], Any, Double)]): TrainingResult = {
    train(trainingData.map(r => (r._1, r._2)), Some(trainingData.map(_._3)))
  }

}

/** A learner that trains on multiple labels, outputting a single model that makes predictions for all labels.
  */
trait MultiTaskLearner extends Serializable {

  /** Train a model
    *
    * @param trainingData
    *   to train on. Each entry is a tuple (vector of inputs, vector of labels)
    * @param weights
    *   for the training rows, if applicable
    * @return
    *   A training result that encompasses model(s) for all labels.
    */
  def train(trainingData: Seq[(Vector[Any], Vector[Any])], weights: Option[Seq[Double]] = None): MultiTaskTrainingResult
}
