package io.citrine.lolo.linear

import breeze.linalg.{diag, pinv, sum, DenseMatrix, DenseVector}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

/** Linear and ridge regression learner
  *
  * Created by maxhutch on 12/6/16.
  *
  * @param fitIntercept
  *   whether to fit an intercept or not
  */
case class LinearRegressionLearner(
    regParam: Option[Double] = None,
    fitIntercept: Boolean = true
) extends Learner {

  /** Train a linear model via direct inversion.
    *
    * @param trainingData
    *   to train on
    * @param weights
    *   for the training rows, if applicable
    * @return
    *   a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]]
  ): LinearRegressionTrainingResult = {
    val n = trainingData.size

    /* Get the indices of the continuous features */
    val indices: Vector[Int] = trainingData.head._1.zipWithIndex
      .filter(_._1.isInstanceOf[Double])
      .filterNot(_._1.asInstanceOf[Double].isNaN)
      .map(_._2)
      .filterNot(i => trainingData.exists(_._1(i).asInstanceOf[Double].isNaN))
      .filterNot { i =>
        val unregularized = !regParam.exists(_.asInstanceOf[Double] > 0.0)
        lazy val constant = trainingData.forall(_._1(i) == trainingData.head._1(i))
        unregularized && constant // remove constant features if there's no regularization
      }

    /* If we are fitting the intercept, add a row of 1s */
    val At = if (fitIntercept) {
      new DenseMatrix(
        indices.size + 1,
        n,
        trainingData.map(r => indices.map(r._1(_).asInstanceOf[Double]) :+ 1.0).flatten.toArray
      )
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

    val b = if (weightsMatrix.isDefined) {
      new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).zip(weights.get).map(p => p._1 * p._2).toArray)
    } else {
      new DenseVector(trainingData.map(_._2.asInstanceOf[Double]).toArray)
    }

    val beta = if (regParam.exists(_ > 0) || n >= k) {
      /* Construct the regularized problem and solve it */
      val regVector = Math.pow(regParam.getOrElse(0.0), 2) * DenseVector.ones[Double](k)
      if (fitIntercept) regVector(-1) = 0.0
      val M = At * A + diag(regVector)
      try {
        val Mi = pinv(M)
        /* Backsub to get the coefficients */
        Mi * At * b
      } catch {
        case x: Throwable =>
          val mean = if (weightsMatrix.isDefined) sum(b) / weights.get.sum else sum(b) / b.length
          val res = DenseVector.zeros[Double](k)
          res(-1) = mean
          res
      }
    } else {
      pinv(A) * b
    }

    val indicesToModel = if (indices.size < trainingData.head._1.size) {
      Some(indices, trainingData.head._1.size)
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

/** Simple container around the model
  *
  * @param model
  *   contained
  */
class LinearRegressionTrainingResult(model: LinearRegressionModel) extends TrainingResult {

  override def getModel(): LinearRegressionModel = model

  /** Get a measure of the importance of the model features
    *
    * @return
    *   feature influences as an array of doubles
    */
  override def getFeatureImportance(): Option[Vector[Double]] = {
    val beta: Vector[Double] = model.getBeta().map(Math.abs)
    val renorm: Double = 1.0 / beta.sum
    Some(beta.map(_ * renorm))
  }
}

/** Linear regression model as a coefficient vector and intercept
  *
  * @param beta
  *   coefficient vector
  * @param intercept
  *   intercept
  * @param indices
  *   optional indices from which to extract real features
  */
class LinearRegressionModel(
    beta: DenseVector[Double],
    intercept: Double,
    indices: Option[(Vector[Int], Int)] = None
) extends Model[LinearRegressionResult] {

  /** Apply the model to a seq of inputs
    *
    * @param inputs
    *   to apply the model to
    * @return
    *   a predictionresult which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): LinearRegressionResult = {
    val filteredInputs = indices
      .map { case (ind, size) => inputs.map(inp => ind.map(inp(_))) }
      .getOrElse(inputs)
      .flatten
      .asInstanceOf[Seq[Double]]
    val inputMatrix = new DenseMatrix(filteredInputs.size / inputs.size, inputs.size, filteredInputs.toArray)
    val resultVector = beta.t * inputMatrix + intercept
    val result = resultVector.t.toArray.toSeq
    val grad = getBeta()
    new LinearRegressionResult(result, grad)
  }

  /** Get the beta from the linear model \beta^T X = y
    * @return
    *   beta as a vector of double
    */
  def getBeta(): Vector[Double] = {
    indices
      .map { case (inds, size) =>
        val empty = DenseVector.zeros[Double](size)
        inds.zipWithIndex.foreach { case (j, i) => empty(j) = beta(i) }
        empty
      }
      .getOrElse(beta)
      .toArray
      .toVector
  }
}

/** Simple container around the result and coefficient array
  *
  * @param values
  *   computed from the model
  * @param grad
  *   gradient vector, which are just the linear coefficients
  */
class LinearRegressionResult(values: Seq[Double], grad: Vector[Double]) extends PredictionResult[Double] {

  /** Get the expected values for this prediction
    *
    * @return
    *   expected value of each prediction
    */
  override def getExpected(): Seq[Double] = values

  /** Get the gradient, which is uniform
    *
    * @return
    *   a vector of doubles for each prediction
    */
  override def getGradient(): Option[Seq[Vector[Double]]] = Some(Seq.fill(values.size)(grad))
}
