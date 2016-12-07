package io.citrine.lolo.linear

import breeze.linalg.{DenseMatrix, DenseVector, det, diag, inv, norm, pinv}
import breeze.numerics.NaN
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
    val indices = trainingData.head._1.zipWithIndex
      .filter(_._1.isInstanceOf[Double])
      .filterNot(_._1.asInstanceOf[Double].isNaN)
      .map(_._2)
      .filterNot(i => trainingData.exists(_._1(i).asInstanceOf[Double].isNaN))


    /* If we are fitting the intercept, add a row of 1s */
    val At = if (fitIntercept) {
      new DenseMatrix(indices.size + 1, n, trainingData.map(r => indices.map(r._1(_).asInstanceOf[Double]) :+ 1.0).flatten.toArray)
    } else {
      new DenseMatrix(indices.size, n, trainingData.map(r => indices.map(r._1(_).asInstanceOf[Double])).flatten.toArray)
    }
    val k = At.rows
    assert(!At.toArray.toSeq.exists(_.isNaN), "There is a NaN in At")

    /* If the weights are specified, multiply At by them */
    val weightsMatrix = weights.map(w => diag(new DenseVector(w.toArray)))
    val Atw = if (weightsMatrix.isDefined) {
      At * weightsMatrix.get
    } else {
      At
    }
    val A = Atw.t
    assert(!A.toArray.toSeq.exists(_.isNaN), "There is a NaN in A")

    val b = if (weightsMatrix.isDefined){
      new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).zip(weights.get).map(p => p._1 * p._2).toArray)
    } else {
      new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).toArray)
    }
    assert(!norm(b).isNaN, s"There is a NaN in b ${weightsMatrix.get.size} ${b.size}:\n ${trainingData.map(_._2)} \n ${weights.get}")

    val beta = if (regParam > 0 || n >= k) {
      /* Construct the regularized problem and solve it */
      val regVector = regParam * regParam * DenseVector.ones[Double](k)
      if (fitIntercept) regVector(-1) = 0.0
      val M = At * A + diag(regVector)
      assert(!M.toArray.toSeq.exists(_.isNaN), "There is a NaN in M")
      val Mi = inv(M)
      assert(!Mi.toArray.toSeq.exists(_.isNaN), "There is a NaN in Mi")
      /* Backsub to get the coefficients */
      Mi * At * b
    } else {
      pinv(A) * b
    }
    assert(!norm(beta).isNaN, "There is a NaN in beta")

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
    assert(!result.exists(_.isNaN), s"Result is NaN: ${result} \n${filteredInputs}\n${intercept} ${beta.toArray.toVector}")
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
