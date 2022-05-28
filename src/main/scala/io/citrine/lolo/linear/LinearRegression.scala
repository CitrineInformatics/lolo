package io.citrine.lolo.linear

import breeze.linalg.svd.SVD
import breeze.linalg.{diag, sum, svd, DenseMatrix, DenseVector}
import breeze.numerics.sqrt
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
    fitIntercept: Boolean = true,
    svdTolerance: Double = 0.0
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
    val alpha = regParam.getOrElse(0.0)
    val regularized = alpha > 0.0

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

    val numSamples = features.length
    val numFeatures = indices.length + (if (fitIntercept) 1 else 0)

    /* Assemble breeze vectors for solving normal equations  */
    val Xt = if (fitIntercept) {
      val featureArray = features.flatMap { row => indices.map(idx => row(idx).asInstanceOf[Double]) :+ 1.0 }.toArray
      new DenseMatrix(numFeatures, numSamples, featureArray)
    } else {
      val featureArray = features.flatMap { row => indices.map(idx => row(idx).asInstanceOf[Double]) }.toArray
      new DenseMatrix(numFeatures, numSamples, featureArray)
    }
    val y = new DenseVector(labels.map(_.asInstanceOf[Double]).toArray)

    /* Rescale data by weight matrix */
    val sqrtWeightsMatrix = weights.map(w => diag(sqrt(new DenseVector(w.toArray))))
    val yw = sqrtWeightsMatrix.map(W => W * y).getOrElse(y)
    val Xtw = sqrtWeightsMatrix.map(W => Xt * W).getOrElse(Xt)
    val Xw = Xtw.t

    val (coefficients, intercept) = Try {
      val (a, b) = if (regularized || numSamples >= numFeatures) {
//        val L = math.pow(alpha, 2) * DenseVector.ones[Double](numFeatures)
//        if (fitIntercept) {
//          L(-1) = 0.0
//        }
        (Xtw * Xw, Xtw * yw)
      } else {
        (Xw, yw)
      }

      val beta = solveSVD(a, b, alpha, svdTolerance)
      //val beta = a \ b

//      val beta = if (regularized || numSamples >= numFeatures) {
//        /* Construct the regularized problem and solve it */
//        val alpha = math.pow(thisRegParam, 2) * DenseVector.ones[Double](numFeatures)
//        if (fitIntercept) {
//          alpha(-1) = 0.0
//        }
//        val A = Xtw * Xw + diag(alpha)
//        val b = Xtw * yw
//        println(solveSVD(A, b, thisRegParam))
//        A \ b
//      } else {
//        /* The problem is under-determined, so use pseudo-inverse as an approximate solution */
//        pinv(Xw) * yw
//      }
      if (fitIntercept) {
        (beta(0 to -2), beta(-1))
      } else {
        (beta, 0.0)
      }
    } getOrElse {
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

    val model = new LinearRegressionModel(coefficients, intercept, indices = indicesToModel)
    new LinearRegressionTrainingResult(model)
  }

  /**
    * Determine beta via SVD with singular value cutoff tolerance.
    *
    * Equations from _solve_svd @ https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py
    */
  private def solveSVD(
      X: DenseMatrix[Double],
      y: DenseVector[Double],
      alpha: Double,
      tol: Double
  ): DenseVector[Double] = {
    val SVD(u, s, vt) = svd(X)

    val Uty = u.t * y
    val D = s.map { si =>
      if (math.abs(si) > tol) {
        si / (si * si + alpha)
      } else {
        0.0
      }
    }
    val D_Uty = D *:* Uty

    // Breeze SVD returns full matrices, we only need min version up to numSamples columns
    val Vm = vt.t(::, 0 until D_Uty.length)
    Vm * D_Uty
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
