package io.citrine.lolo.linear

import breeze.linalg.{det, diag, pinv, sum, DenseMatrix, DenseVector}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}

import scala.util.Try

/**
  * Linear and ridge regression learner
  *
  * Created by maxhutch on 12/6/16.
  *
  * @param fitIntercept whether to fit an intercept or not
  */
case class LinearRegressionLearner(
    regParam: Option[Double] = None,
    fitIntercept: Boolean = true
) extends Learner {

  /**
    * Train a linear model via direct inversion.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @return a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Any)],
      weights: Option[Seq[Double]]
  ): LinearRegressionTrainingResult = {
    val thisRegParam = regParam.getOrElse(0.0)
    val regularized = thisRegParam > 0.0

    val theseWeights = weights.getOrElse(Seq.fill(trainingData.length)(1.0))
    val (nonZeroData, nonZeroWeights) = trainingData.zip(theseWeights).filter(_._2 > 0.0).unzip

    println(nonZeroWeights)

    val n = nonZeroData.size
    val (features, labels) = nonZeroData.unzip
    val rep = features.head

    /* Get the indices of the continuous features */
    val indices = rep.zipWithIndex
      .collect { case (x: Double, idx) if !x.isNaN => idx }
      .filterNot { idx =>
        // Remove features with NaN values in training set
        val hasNans = features.exists(row => row(idx).asInstanceOf[Double].isNaN)
        // Remove constant features if there's no regularization
        lazy val constant = features.forall(row => row(idx) == rep(idx))
        hasNans || (!regularized && constant)
      }

    /* If we are fitting the intercept, add a row of 1s */
    val Xt = if (fitIntercept) {
      val featureArray = features.flatMap { row => indices.map(idx => row(idx).asInstanceOf[Double]) :+ 1.0 }.toArray
      new DenseMatrix(indices.size + 1, n, featureArray)
    } else {
      val featureArray = features.flatMap { row => indices.map(idx => row(idx).asInstanceOf[Double]) }.toArray
      new DenseMatrix(indices.size, n, featureArray)
    }
    val X = Xt.t
    val k = Xt.rows

    /* Multiply Xt by weight matrix */
    val Xtw = Xt * diag(new DenseVector(nonZeroWeights.toArray))

    val y = new DenseVector(labels.map(_.asInstanceOf[Double]).toArray)

    val beta = if (regularized || n >= k) {
      /* Construct the regularized problem and solve it */
      val regVector = math.pow(thisRegParam, 2) * DenseVector.ones[Double](k)
      if (fitIntercept) {
        regVector(-1) = 0.0
      }
      val lhs = Xtw * X + diag(regVector)
      val rhs = Xtw * y
      Try {
        lhs \ rhs
      } getOrElse {
        val totalWeight = nonZeroWeights.sum
        val mean = sum(y) / totalWeight
        val res = DenseVector.zeros[Double](k)
        res(-1) = mean
        res
      }
    } else {
      /* The problem is under-determined, so use pinv as approximate solution */
      val Xw = Xtw.t
      pinv(Xw) * y
    }

    val indicesToModel = if (indices.size < rep.size) {
      Some(indices, trainingData.head._1.size)
    } else {
      None
    }

    /* If we fit the intercept, take it off the end of the coefficients */
    val model = if (fitIntercept) {
      new LinearRegressionModel(beta(0 to -2), beta(-1), indices = indicesToModel)
    } else {
      new LinearRegressionModel(beta, intercept = 0.0, indices = indicesToModel)
    }

    new LinearRegressionTrainingResult(model)
  }
}

/**
  * Simple container around the model
  *
  * @param model contained
  */
class LinearRegressionTrainingResult(model: LinearRegressionModel) extends TrainingResult {

  override def getModel(): LinearRegressionModel = model

  /**
    * Get a measure of the importance of the model features
    *
    * @return feature influences as an array of doubles
    */
  override def getFeatureImportance(): Option[Vector[Double]] = {
    val beta: Vector[Double] = model.getBeta().map(Math.abs)
    val renorm: Double = 1.0 / beta.sum
    Some(beta.map(_ * renorm))
  }
}

/**
  * Linear regression model as a coefficient vector and intercept
  *
  * @param beta      coefficient vector
  * @param intercept intercept
  * @param indices   optional indices from which to extract real features
  */
class LinearRegressionModel(
    beta: DenseVector[Double],
    intercept: Double,
    indices: Option[(Vector[Int], Int)] = None
) extends Model[LinearRegressionResult] {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a predictionresult which includes, at least, the expected outputs
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

  /**
    * Get the beta from the linear model \beta^T X = y
    * @return beta as a vector of double
    */
  def getBeta(): Vector[Double] = {
    indices
      .map {
        case (inds, size) =>
          val empty = DenseVector.zeros[Double](size)
          inds.zipWithIndex.foreach { case (j, i) => empty(j) = beta(i) }
          empty
      }
      .getOrElse(beta)
      .toScalaVector
  }
}

/**
  * Simple container around the result and coefficient array
  *
  * @param values computed from the model
  * @param grad   gradient vector, which are just the linear coefficients
  */
class LinearRegressionResult(values: Seq[Double], grad: Vector[Double]) extends PredictionResult[Double] {

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
  override def getGradient(): Option[Seq[Vector[Double]]] = Some(Seq.fill(values.size)(grad))
}
