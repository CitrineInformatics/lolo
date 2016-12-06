package io.citrine.lolo.linear

import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, pinv}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult, hasGradient}

/**
  * Linear and ridge regression learner
  *
  * Created by maxhutch on 12/6/16.
  *
  * @param regParam for ridge regression.  default = 0
  * @param fitIntercept whether to fit an intercept or not
  */
class LinearRegressionLearner(regParam: Double = 0.0, fitIntercept: Boolean = true) extends Learner {

  /**
    * Train a linear model via direct inversion.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a model
    */
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]]): LinearRegressionTrainingResult = {
    val n = trainingData.size

    /* Get the indices of the continuous features */
    val indices = trainingData.head._1.zipWithIndex.filter(_._1.isInstanceOf[Double]).map(_._2)

    /* If we are fitting the intercept, add a row of 1s */
    val At = if (fitIntercept) {
      new DenseMatrix(indices.size + 1, n, trainingData.map(r => indices.map(r._1(_).asInstanceOf[Double]) :+ 1.0).flatten.toArray)
    } else {
      new DenseMatrix(indices.size, n, trainingData.map(r => indices.map(r._1(_).asInstanceOf[Double])).flatten.toArray)
    }
    val k = At.rows

    /* If the weights are specified, multiply At by them */
    val weightsMatrix = weights.map(w => diag(new DenseVector(w.toArray)))
    val Atw = if (weightsMatrix.isDefined) {
      At * weightsMatrix.get
    } else {
      At
    }
    val A = Atw.t
    val b = if (weightsMatrix.isDefined){
      weightsMatrix.get * new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).toArray)
    } else {
      new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).toArray)
    }

    val beta = if (regParam > 0 || n >= k) {
      /* Construct the regularized problem and solve it */
      val M = At * A + diag(regParam * regParam * DenseVector.ones[Double](k))
      val Mi = inv(M)
      /* Backsub to get the coefficients */
      Mi * At * b
    } else {
      pinv(A) * b
    }

    val indicesToModel = if (indices.size < trainingData.head._1.size) {
      Some(indices)
    } else {
      None
    }

    /* If we fit the intercept, take it off the end of the coefficients */
    val model = if (fitIntercept) {
      new LinearRegressionModel(beta(0 to -2), beta(-1), indices = indicesToModel)
    } else {
      new LinearRegressionModel(beta, 0.0, indices = indicesToModel)
    }

    new LinearRegressionTrainingResult(model)
  }
}

/**
  * Simple container around the model
  * @param model contained
  */
class LinearRegressionTrainingResult(model: LinearRegressionModel) extends TrainingResult {
  override def getModel(): LinearRegressionModel = model
}

/**
  * Linear regression model as a coefficient vector and intercept
  * @param beta coefficient vector
  * @param intercept intercept
  * @param indices optional indices from which to extract real features
  */
class LinearRegressionModel(beta: DenseVector[Double], intercept: Double, indices: Option[Vector[Int]] = None) extends Model {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): LinearRegressionResult = {
    val filteredInputs = indices.map(ind => inputs.map(inp => ind.map(inp(_)))).getOrElse(inputs).flatten.asInstanceOf[Seq[Double]]
    val inputMatrix = new DenseMatrix(filteredInputs.size / inputs.size, inputs.size,
      filteredInputs.toArray
    )
    val resultVector = beta.t * inputMatrix + intercept
    val result = resultVector.t.toArray.toSeq
    val grad = indices.map{inds =>
      val empty = DenseVector.zeros[Double](inputs.head.size)
      inds.zipWithIndex.foreach{ case (j, i) => empty(j) = beta(i)}
      empty
    }.getOrElse(beta).toArray.toVector
    new LinearRegressionResult(result, grad)
  }
}

/**
  * Simple container around the result and coefficient array
  * @param values computed from the model
  * @param grad gradient vector, which are just the linear coefficients
  */
class LinearRegressionResult(values: Seq[Double], grad: Vector[Double]) extends PredictionResult with hasGradient {
  /**
    * Get the expected values for this prediction
    *
    * @return expected value of each prediction
    */
  override def getExpected(): Seq[Double] = values

  /**
    * Get the gradient, which is uniform
    *
    * @return a vector of doubles for each prediction
    */
  override def getGradient(): Seq[Vector[Double]] = Seq.fill(values.size)(grad)
}
