package io.citrine.lolo.linear

import breeze.linalg.{norm, DenseMatrix, DenseVector}
import io.citrine.lolo.SeedRandomMixIn
import io.citrine.lolo.api.TrainingRow
import org.junit.Test

/**
  * Created by maxhutch on 12/6/16.
  */
@Test
class LinearRegressionTest extends SeedRandomMixIn {

  /** Number of training rows */
  val n = 6

  /** Number of features in each row */
  val k = 4

  /** Generate random training data */
  val data = DenseMatrix.rand[Double](n, k)

  /** And a random model */
  val beta0 = DenseVector.rand[Double](k)

  /**
    * Test regression without an intercept or weight; this is easiest
    */
  @Test
  def testRegressionNoIntercept(): Unit = {
    val result = data * beta0

    val trainingData = (0 until n).map { i =>
      TrainingRow(data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = LinearRegressionLearner(fitIntercept = false)
    val lrm = lr.train(trainingData)
    val model = lrm.model
    val output = model.transform(trainingData.map(_.inputs))
    val predicted = output.expected
    val beta = output.gradient.get.head

    assert(norm(new DenseVector(beta.toArray) - beta0) < 1.0e-9, "Coefficients are inaccurate")
    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Add an intercept
    */
  @Test
  def testRegression(): Unit = {
    val int0 = rng.nextDouble()
    val result = data * beta0 + int0

    val trainingData = (0 until n).map { i =>
      TrainingRow(data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = LinearRegressionLearner()
    val lrm = lr.train(trainingData)
    val model = lrm.model
    val output = model.transform(trainingData.map(_.inputs))
    val predicted = output.expected
    val beta = output.gradient.get.head

    assert(norm(new DenseVector(beta.toArray) - beta0) < 1.0e-9, "Coefficients are inaccurate")
    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Add an intercept and random weights
    */
  @Test
  def testWeightedRegression(): Unit = {
    val int0 = rng.nextDouble()
    val result = data * beta0 + int0

    val trainingData = (0 until n).map { i =>
      TrainingRow(data.t(::, i).toDenseVector.toArray.toVector, result(i), math.abs(rng.nextDouble()))
    }

    val lr = LinearRegressionLearner()
    val lrm = lr.train(trainingData, rng = rng)
    val model = lrm.model
    val output = model.transform(trainingData.map(_.inputs))
    val predicted = output.expected
    val beta = output.gradient.get.head

    assert(norm(new DenseVector(beta.toArray) - beta0) < 1.0e-9, "Coefficients are inaccurate")
    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Test when there are more features than training rows
    */
  @Test
  def testUnderconstrained(): Unit = {

    /** Number of training rows */
    val n = 4

    /** Number of features in each row */
    val k = 6

    /** Generate random training data */
    val data = DenseMatrix.rand[Double](n, k)

    /** And a random model */
    val beta0 = DenseVector.rand[Double](k)
    val result = data * beta0

    val trainingData = (0 until n).map { i =>
      TrainingRow(data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = LinearRegressionLearner(fitIntercept = false)
    val lrm = lr.train(trainingData)
    val model = lrm.model
    val output = model.transform(trainingData.map(_.inputs))
    val predicted = output.expected
    val beta = output.gradient.head

    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Add an intercept
    */
  @Test
  def testCategoricalValue(): Unit = {
    val int0 = rng.nextDouble()
    val result = data * beta0 + int0

    val trainingData = (0 until n).map { i =>
      val inputs = "Foo" +: data.t(::, i).toDenseVector.toArray.toVector.asInstanceOf[Vector[Any]] :+ "Bar"
      TrainingRow(inputs, result(i))
    }

    val lr = LinearRegressionLearner()
    val lrm = lr.train(trainingData)
    val model = lrm.model
    val output = model.transform(trainingData.map(_.inputs))
    val predicted = output.expected
    val beta = output.gradient.get.head
    val importance = lrm.featureImportance.get

    /* Make sure that feature importance matches the gradient */
    val betaScale = beta.map(Math.abs).sum
    beta.zip(importance).foreach {
      case (b, i) =>
        val diff = Math.abs(Math.abs(b / betaScale) - i)
        assert(
          diff < Double.MinPositiveValue || diff / i < 1.0e-9,
          s"Beta and feature importance disagree: ${b / betaScale} vs ${i}"
        )
    }

    assert(norm(new DenseVector(beta.toArray)(1 to -2) - beta0) < 1.0e-9, "Coefficients are inaccurate")
    assert(beta.head == 0.0, "Non-zero graident given to leading categorical feature")
    assert(beta.last == 0.0, "Non-zero graident given to trailing categorical feature")
    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

}

/** Companion driver */
object LinearRegressionTest {

  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    new LinearRegressionTest().testRegressionNoIntercept()
    new LinearRegressionTest().testRegression()
    new LinearRegressionTest().testWeightedRegression()
    new LinearRegressionTest().testUnderconstrained()
    new LinearRegressionTest().testCategoricalValue()
  }
}
