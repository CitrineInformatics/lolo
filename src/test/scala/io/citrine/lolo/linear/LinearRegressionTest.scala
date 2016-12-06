package io.citrine.lolo.linear

import breeze.linalg.{DenseMatrix, DenseVector}
import org.junit.Test

/**
  * Created by maxhutch on 12/6/16.
  */
@Test
class LinearRegressionTest {

  def testRegressionNoIntercept(): Unit = {
    val n = 5
    val k = 4
    val data = DenseMatrix.rand[Double](n,k)
    val beta0 = DenseVector.rand[Double](k)
    val result = data * beta0

    val trainingData = (0 until n).map{ i =>
      (data.t(::, i).toDenseVector.toArray.toVector, result(i))
    }

    val lr = new LinearRegressionLearner()
    val lrm = lr.train(trainingData)
    val model = lrm.getModel()
    val output = model.transform(trainingData.map(_._1))
    val predicted = output.getExpected()

    val beta = output.getGradient().head
    println(beta, beta0.toArray.toVector)

    predicted.zip(result.toArray).foreach{ case (p, a) =>
      println(s"Predicted vs Actual: ${p}, ${a}")
      assert(Math.abs(p - a) < 1.0e-9)
    }

  }
}

object LinearRegressionTest {
  def main(argv: Array[String]): Unit = {
    new LinearRegressionTest().testRegressionNoIntercept()
  }
}
