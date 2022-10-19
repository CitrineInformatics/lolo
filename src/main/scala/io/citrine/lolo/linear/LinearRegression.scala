package io.citrine.lolo.linear

import breeze.linalg.{diag, sum, DenseMatrix, DenseVector}
import io.citrine.lolo.{Learner, Model, PredictionResult, TrainingResult}
import io.citrine.random.Random
import org.slf4j.LoggerFactory

import scala.util.{Failure, Success, Try}

/**
  * Linear ridge regression learner.
  *
  * @param regParam regularization to use for the ridge model
  * @param fitIntercept whether to fit an intercept or not
  */
case class LinearRegressionLearner(
    regParam: Option[Double] = None,
    fitIntercept: Boolean = true
) extends Learner[Double] {

  /**
    * Train a linear model via direct inversion.
    *
    * @param trainingData to train on
    * @param weights      for the training rows, if applicable
    * @param rng          random number generator for reproducibility
    * @return a model
    */
  override def train(
      trainingData: Seq[(Vector[Any], Double)],
      weights: Option[Seq[Double]],
      rng: Random
  ): LinearRegressionTrainingResult = {
    val lambda = regParam.getOrElse(0.0)
    val regularized = lambda > 0.0

    val (features, labels) = trainingData.unzip
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

    val numSamples = trainingData.length
    val numFeatures = indices.length + (if (fitIntercept) 1 else 0)

    /* Assemble breeze vectors for solving normal equations  */
    val Xt = if (fitIntercept) {
      val featureArray = features.flatMap { row => indices.map(idx => row(idx).asInstanceOf[Double]) :+ 1.0 }.toArray
      new DenseMatrix(numFeatures, numSamples, featureArray)
    } else {
      val featureArray = features.flatMap { row => indices.map(idx => row(idx).asInstanceOf[Double]) }.toArray
      new DenseMatrix(numFeatures, numSamples, featureArray)
    }
    val X = Xt.t
    val y = new DenseVector(labels.toArray)

    /* Rescale data by weight matrix */
    val weightsMatrix = weights.map(w => diag(new DenseVector(w.toArray)))
    val Xw = weightsMatrix.map(W => W * X).getOrElse(X)
    val yw = weightsMatrix.map(W => W * y).getOrElse(y)

    val (coefficients, intercept) = Try {
      // For a regularized/overdetermined problem, the LHS operator `A` is (potentially) invertible
      // and can be applied directly to solve the normal equations (using an LU/QR internally)
      // When numFeatures > numSamples, the operator `A` is singular and cannot be inverted directly
      // In this case, we fall back to a least-squares solution calling `\` directly on `Xw` and `yw`
      if (regularized || numSamples >= numFeatures) {
        val l = math.pow(lambda, 2) * DenseVector.ones[Double](numFeatures)
        if (fitIntercept) l(-1) = 0.0
        val A = Xt * Xw + diag(l)
        val b = Xt * yw
        A \ b
      } else {
        Xw \ yw
      }
    } match {
      case Success(beta) =>
        if (fitIntercept) {
          (beta(0 to -2), beta(-1))
        } else {
          (beta, 0.0)
        }
      case Failure(e) =>
        logger.warn(s"Encountered an exception solving normal equations: ${e.getLocalizedMessage}")
        val totalWeight = weights.map(_.sum).getOrElse(numSamples.toDouble)
        val mean = sum(yw) / totalWeight // weighted mean of training labels
        if (fitIntercept) {
          (DenseVector.zeros[Double](numFeatures - 1), mean)
        } else {
          (DenseVector.zeros[Double](numFeatures), mean)
        }
    }

    /* Extract active indices for trained model */
    val indicesToModel = if (indices.length < rep.length) {
      Some(indices, rep.length)
    } else {
      None
    }

    val model = LinearRegressionModel(coefficients, intercept, indices = indicesToModel)
    LinearRegressionTrainingResult(model)
  }

  private val logger = LoggerFactory.getLogger(getClass)
}

/**
  * Simple container around the model
  *
  * @param model contained
  */
case class LinearRegressionTrainingResult(model: LinearRegressionModel) extends TrainingResult[Double] {

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
case class LinearRegressionModel(
    beta: DenseVector[Double],
    intercept: Double,
    indices: Option[(Vector[Int], Int)] = None
) extends Model[Double] {

  /**
    * Apply the model to a seq of inputs
    *
    * @param inputs to apply the model to
    * @return a prediction result which includes, at least, the expected outputs
    */
  override def transform(inputs: Seq[Vector[Any]]): LinearRegressionResult = {
    val filteredInputs = indices
      .map { case (ind, _) => inputs.map(inp => ind.map(inp(_))) }
      .getOrElse(inputs)
      .flatten
      .asInstanceOf[Seq[Double]]
    val inputMatrix = new DenseMatrix(filteredInputs.size / inputs.size, inputs.size, filteredInputs.toArray)
    val resultVector = beta.t * inputMatrix + intercept
    val result = resultVector.t.toArray.toSeq
    val grad = getBeta()
    LinearRegressionResult(result, grad)
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
case class LinearRegressionResult(values: Seq[Double], grad: Vector[Double]) extends PredictionResult[Double] {

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
