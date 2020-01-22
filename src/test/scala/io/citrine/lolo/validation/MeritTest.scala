package io.citrine.lolo.validation

import io.citrine.lolo.PredictionResult
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.{Random, Try}

class MeritTest {

  /**
    * Generate test data by adding Gaussian noise to a uniformly distributed response
    *
    * Uncertainty estimates are also produced.  The degree of correlation between the uncertainty estimate and the
    * error is configurable.
    *
    * @param noiseScale             the scale of the errors added to the responses
    * @param uncertaintyCorrelation the degree of correlation between the predicted uncertainty and the local error scale
    * @param batchSize              the number of predictions per result
    * @param numBatch               the number of prediction results produced
    * @return predicted-vs-actual data in the format expected by Metric.estimate
    */
  private def getNormalPVA(
                            noiseScale: Double = 1.0,
                            uncertaintyCorrelation: Double = 0.0,
                            batchSize: Int = 32,
                            numBatch: Int = 1
                          ): Iterable[(PredictionResult[Double], Seq[Double])] = {
    val maximumCorrelation = 0.999

    val noiseVariance = noiseScale * noiseScale
    val noiseUncertaintyCovariance = noiseVariance * Math.min(uncertaintyCorrelation, maximumCorrelation) // avoid singular matrices
    val errorDistribution = new MultivariateNormalDistribution(
      Array(0.0, 0.0),
      Array(Array(noiseVariance, noiseUncertaintyCovariance), Array(noiseUncertaintyCovariance, noiseVariance))
    )

    Seq.fill(numBatch) {
      val pua = Seq.fill(batchSize) {
        val y = Random.nextDouble()
        val draw = errorDistribution.sample().toSeq
        val error: Double = draw(0) * Random.nextGaussian()
        val uncertainty = if (uncertaintyCorrelation >= maximumCorrelation) {
          Math.abs(draw(0))
        } else {
          Math.abs(draw(1))
        }
        (y + error, uncertainty, y)
      }
      val predictionResult = new PredictionResult[Double] {
        override def getExpected(): Seq[Double] = pua.map(_._1)

        override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = Some(pua.map(_._2))
      }
      (predictionResult, pua.map(_._3))
    }
  }

  /**
    * Test that the RMSE comes out correctly
    *
    * This has an false negative rate below 1/100
    */
  @Test
  def testRMSE(): Unit = {
    val pva = getNormalPVA(batchSize = 256, numBatch = 32)
    val (rmse, uncertainty) = RootMeanSquareError.estimate(pva)
    assert(Math.abs(rmse - 1.0) < 3 * uncertainty, "RMSE estimate was not accurate enough")
    assert(uncertainty < 0.05, s"RMSE estimate was not precise enough")
  }

  /**
    * Test that the coefficient of determination comes out right
    *
    * This has an false negative rate below 1/100
    */
  @Test
  def testCoefficientOfDeterminization(): Unit = {
    val pva = getNormalPVA(noiseScale = 0.1, batchSize = 256, numBatch = 32)
    val expected = 1 - 12 * 0.1 * 0.1
    val (r2, uncertainty) = CoefficientOfDetermination.estimate(pva)
    assert(Math.abs(r2 - expected) < 3 * uncertainty, "R^2 estimate was not accurate enough")
    assert(uncertainty < 0.05, s"R^2 estimate was not precise enough")
  }

  /**
    * Test that the standard confidence comes out right
    *
    * This has an false negative rate below 1/200
    */
  @Test
  def testStandardConfidence(): Unit = {
    val pva = getNormalPVA(uncertaintyCorrelation = 1.0, batchSize = 256, numBatch = 32)
    val expected = 0.68
    val (confidence, uncertainty) = StandardConfidence.estimate(pva)
    assert(Math.abs(confidence - expected) < 3 * uncertainty, "Confidence estimate was not accurate enough")
    assert(uncertainty < 0.05, s"Confidence estimate was not precise enough")
  }

  /**
    * Test that the standard error comes out right
    *
    * This has an false negative rate below 1/100
    */
  @Test
  def testStandardError(): Unit = {
    val pva = getNormalPVA(noiseScale = 0.01, uncertaintyCorrelation = 1.0, batchSize = 256, numBatch = 32)
    val expected = 1.0
    val (error, uncertainty) = StandardError().estimate(pva)
    assert(Math.abs(error - expected) < 3 * uncertainty, "Standard error estimate was not accurate enough")
    assert(uncertainty < 0.05, s"Standard error estimate was not precise enough")
  }


  /**
    * Test that the UncertaintyCorrelation is correct when perfectly correlated
    *
    * This has an false negative rate below 1/100
    */
  @Test
  def testPerfectUncertaintyCorrelation(): Unit = {
    val pva = getNormalPVA(noiseScale = 0.01, uncertaintyCorrelation = 1.0, batchSize = 256, numBatch = 32)
    val expected = 1.0
    val (corr, uncertainty) = UncertaintyCorrelation.estimate(pva)
    assert(Math.abs(corr - expected) < 3 * uncertainty, "Uncertainty correlation estimate was not accurate enough")
    assert(uncertainty < 0.05, s"Uncertainty correlation estimate was not precise enough")
  }

  /**
    * Test that the UncertaintyCorrelation is correct when there is no correlation
    *
    * This has an false negative rate below 1/100
    */
  @Test
  def testZeroUncertaintyCorrelation(): Unit = {
    val pva = getNormalPVA(noiseScale = 0.01, uncertaintyCorrelation = 0.0, batchSize = 256, numBatch = 32)
    val expected = 0.0
    val (corr, uncertainty) = UncertaintyCorrelation.estimate(pva)
    assert(Math.abs(corr - expected) < 3 * uncertainty, "Uncertainty correlation estimate was not accurate enough")
    assert(uncertainty < 0.05, s"Uncertainty correlation estimate was not precise enough")
  }
}

object MeritTest {
  /**
    * Driver to test tolerance calibration so the false negative rate is low
    */
  def main(args: Array[String]): Unit = {
    val N = 1024
    val failures = Seq.fill(N) {
      Try(
        new MeritTest().testZeroUncertaintyCorrelation() // place test here
      ).isSuccess
    }.count(!_)
    println(s"Failure rate is ${failures.toDouble / N}")
  }
}
