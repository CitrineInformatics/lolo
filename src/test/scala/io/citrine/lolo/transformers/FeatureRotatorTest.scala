package io.citrine.lolo.transformers

import breeze.linalg.{DenseMatrix, DenseVector, det}
import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import org.junit.Test

import scala.util.Random

/**
  * Created by gregor-robinson on 2020-01-06.
  */
@Test
class FeatureRotatorTest {

  val data: Vector[(Vector[Double], Double)] = TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)
  val weights: Vector[Double] = Vector.fill(data.size)(if (Random.nextBoolean()) Random.nextDouble() else 0.0)

  // Creating another dataset which has 1 feature that has 0 variance.
  val dataWithConstant: Vector[(Vector[Double], Double)] = data.map(d => (0.0 +: d._1, d._2))

  @Test
  def testRandomRotation(): Unit = {
    val inputs = data.map(_._1)
    for (i <- 1 to 10) {
      val U = FeatureRotator.getRandomRotation(inputs.head.length)
      assert(U.rows == inputs.head.length)
      assert(U.cols == inputs.head.length)

      // Check that the determinant is one.
      val d = det(U)

      // Check that the matrix is unitary.
      (U.t * U - DenseMatrix.eye[Double](inputs.head.length)).toArray.foreach{ x =>
        assert(Math.abs(x) < 1e-9)
      }

      val featuresToRotate = inputs.head.indices.asInstanceOf[IndexedSeq[Int]]
      val rotatedInputs = FeatureRotator.applyRotation(inputs, featuresToRotate, U)
    }
  }

  /**
    * Guess the mean should be invariant under rotation.
    */
  @Test
  def testStandardGTM(): Unit = {
    val learner = GuessTheMeanLearner()
    val model = learner.train(data).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val rotatedLearner = FeatureRotator(GuessTheMeanLearner())
    val rotatedModel = rotatedLearner.train(data).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1)).getExpected()

    result.zip(rotatedResult).foreach { case (free: Double, rotated: Double) =>
      assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }
  }

  /**
    * Linear regression should be invariant under standardization
    */
  @Test
  def testStandardLinear(): Unit = {
    val learner = LinearRegressionLearner()
    val model = learner.train(data, Some(weights)).getModel()
    val result = model.transform(data.map(_._1))
    val expected = result.getExpected()
    val gradient = result.getGradient()

    val rotatedLearner = FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data, Some(weights)).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1))
    val rotatedExpected = rotatedResult.getExpected()
    val rotatedGradient = rotatedResult.getGradient()

    expected.zip(rotatedExpected).foreach { case (free: Double, rotated: Double) =>
      assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }

    // TODO: test gradient.
    //gradient.get.zip(rotatedGradient.get).foreach { case (free, rotated) =>
    //  val diff = free.zip(rotated).map { case (f, s) => Math.abs(f - s) }.max
    //  assert(diff < 1.0e-9, "Gradients should be the same")
    //}
  }

  /**
    * Ridge regression should not depend on standardization
    */
  @Test
  def testStandardRidge(): Unit = {
    val learner = LinearRegressionLearner(regParam = Some(1.0))
    val model = learner.train(data).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1)).getExpected()

    result.zip(rotatedResult).foreach { case (free: Double, rotated: Double) =>
      assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }
  }

  /*
   * TODO: test with RegressionTreeLearner
   * TODO: test ClassificationTreeLearner
   * TODO: test MultiTaskTreeLearner
   * TODO: test skips non-double input features
   * TODO: investigate performance relative to un-rotated random forest
   */
}
