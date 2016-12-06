package io.citrine.lolo.linear

import breeze.linalg.{DenseMatrix, DenseVector, diag, inv}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult, hasGradient}

/**
  * Created by maxhutch on 12/6/16.
  */
class LinearRegressionLearner(regParam: Double = 0.0, intercept: Boolean = true) extends Learner {
  /**
    * Train a model
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): LinearRegressionTrainingResult = {

    val n = trainingData.size
    val At = if (intercept) {
      new DenseMatrix(trainingData.head._1.size + 1, n, trainingData.map(_._1 :+ 1.0).asInstanceOf[Seq[Vector[Double]]].flatten.toArray)
    } else {
      new DenseMatrix(trainingData.head._1.size, n, trainingData.map(_._1).asInstanceOf[Seq[Vector[Double]]].flatten.toArray)
    }
    val k = At.rows
    val A = At.t

    val M = At * A + diag(regParam * regParam * DenseVector.ones[Double](k))
    val Mi = inv(M)
    val b = new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).toArray)
    val beta = Mi * At * b


    val model = if (intercept) {
      new LinearRegressionModel(beta.toArray.toVector.init, beta(-1))
    } else {
      new LinearRegressionModel(beta.toArray.toVector, 0.0)
    }
    new LinearRegressionTrainingResult(model)
  }
}

class LinearRegressionTrainingResult(model: LinearRegressionModel) extends TrainingResult {
  override def getModel(): LinearRegressionModel = model
}

class LinearRegressionModel(beta: Vector[Double], intercept: Double) extends Model {
  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): LinearRegressionResult = {
    val result = inputs.asInstanceOf[Seq[Vector[Double]]].map(v => v.zip(beta).map(p => p._1 * p._2).sum + intercept)
    new LinearRegressionResult(result, beta)
  }
}

class LinearRegressionResult(values: Seq[Double], grad: Vector[Double]) extends PredictionResult with hasGradient {
  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = values

  /**
    * Get the gradient or sensitivity of each prediction
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Seq[Vector[Double]] = Seq.fill(values.size)(grad)
}
