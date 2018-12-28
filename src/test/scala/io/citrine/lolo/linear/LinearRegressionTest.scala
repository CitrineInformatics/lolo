package io.citrine.lolo.linear

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 12/6/16.
  */
@Test
class LinearRegressionTest {
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
      (data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = LinearRegressionLearner(fitIntercept = false)
    val lrm = lr.train(trainingData)
    val model = lrm.getModel()
    val output = model.transform(trainingData.map(_._1))
    val predicted = output.getExpected()
    val beta = output.getGradient().get.head

    assert(norm(new DenseVector(beta.toArray) - beta0) < 1.0e-9, "Coefficients are inaccurate")
    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Add an intercept
    */
  @Test
  def testRegression(): Unit = {
    val int0 = Random.nextDouble()
    val result = data * beta0 + int0

    val trainingData = (0 until n).map { i =>
      (data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = LinearRegressionLearner()
    val lrm = lr.train(trainingData)
    val model = lrm.getModel()
    val output = model.transform(trainingData.map(_._1))
    val predicted = output.getExpected()
    val beta = output.getGradient().get.head

    assert(norm(new DenseVector(beta.toArray) - beta0) < 1.0e-9, "Coefficients are inaccurate")
    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Add an intercept and random weights
    */
  @Test
  def testWeightedRegression(): Unit = {
    val int0 = Random.nextDouble()
    val result = data * beta0 + int0

    val trainingData = (0 until n).map { i =>
      (data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = new LinearRegressionLearner()
    val lrm = lr.train(trainingData, weights = Some(Seq.tabulate(n)(i => Math.abs(Random.nextDouble()))))
    val model = lrm.getModel()
    val output = model.transform(trainingData.map(_._1))
    val predicted = output.getExpected()
    val beta = output.getGradient().get.head

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
      (data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = new LinearRegressionLearner(fitIntercept = false)
    val lrm = lr.train(trainingData)
    val model = lrm.getModel()
    val output = model.transform(trainingData.map(_._1))
    val predicted = output.getExpected()
    val beta = output.getGradient().head

    assert(norm(new DenseVector(predicted.toArray) - result) < 1.0e-9, "Predictions are inaccurate")
  }

  /**
    * Add an intercept
    */
  @Test
  def testCategoricalValue(): Unit = {
    val int0 = Random.nextDouble()
    val result = data * beta0 + int0

    val trainingData = (0 until n).map { i =>
      ("Foo" +: data.t(::, i).toDenseVector.toArray.toVector.asInstanceOf[Vector[Any]] :+ "Bar", result(i))
    }

    val lr = new LinearRegressionLearner()
    val lrm = lr.train(trainingData)
    val model = lrm.getModel()
    val output = model.transform(trainingData.map(_._1))
    val predicted = output.getExpected()
    val beta = output.getGradient().get.head
    val importance = lrm.getFeatureImportance().get

    /* Make sure that feature importance matches the gradient */
    val betaScale = beta.map(Math.abs).sum
    beta.zip(importance).foreach { case (b, i) =>
      val diff = Math.abs(Math.abs(b / betaScale) - i)
      assert(diff < Double.MinPositiveValue || diff / i < 1.0e-9,
        s"Beta and feature importance disagree: ${b / betaScale} vs ${i}")
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
