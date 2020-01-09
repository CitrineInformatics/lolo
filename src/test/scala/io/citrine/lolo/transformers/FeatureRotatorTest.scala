package io.citrine.lolo.transformers

import breeze.linalg.{DenseMatrix, DenseVector, det}
import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test

import scala.util.Random

/**
  * Created by gregor-robinson on 2020-01-06.
  */
@Test
class FeatureRotatorTest {

  val data: Seq[(Vector[Any], Any)] = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
    inputBins = Seq((0, 8))
  )
  val weights: Vector[Double] = Vector.fill(data.size)(if (Random.nextBoolean()) Random.nextDouble() else 0.0)

  @Test
  def testRandomRotation(): Unit = {
    val inputs = data.map(_._1)
    val numRealInputs = inputs.head.length -1
    for (i <- 1 to 10) {
      val U = FeatureRotator.getRandomRotation(numRealInputs)
      assert(U.rows == numRealInputs)
      assert(U.cols == numRealInputs)

      // Check that the matrix is unitary.
      (U.t * U - DenseMatrix.eye[Double](numRealInputs)).toArray.foreach{ x =>
        assert(Math.abs(x) < 1e-9)
      }

      // Check that the determinant is one.
      val d = det(U)
      assert(Math.abs(d - 1.0) < 1e-9, s"Determinant of U should be 1 but is ${d}")
    }
  }

  @Test
  def testApplyRotation(): Unit = {
    val inputs = data.map(_._1)
    val numRealInputs = inputs.head.length -1
    val featuresToRotate = (1 to numRealInputs).asInstanceOf[IndexedSeq[Int]]

    // Create rotation matrix that exchanges first, second, and last real inputs
    val U = DenseMatrix.eye[Double](dim = numRealInputs)
    U(0,0) = 0.0
    U(1,1) = 0.0
    U(numRealInputs-1,numRealInputs-1) = 0.0
    U(0,1) = 1.0
    U(1,numRealInputs-1) = 1.0
    U(numRealInputs-1,0) = 1.0

    val rotatedInputs = FeatureRotator.applyRotation(inputs, featuresToRotate, U)
    inputs.indices.foreach { i =>
      assert(inputs(i)(0) == rotatedInputs(i)(0), "Failed to leave categorical invariant.")
      assert(Math.abs(inputs(i)(1).asInstanceOf[Double] - rotatedInputs(i)(numRealInputs).asInstanceOf[Double]) < 1e-9, "Failed to exchange coordinates.")
      assert(Math.abs(inputs(i)(2).asInstanceOf[Double] - rotatedInputs(i)(1).asInstanceOf[Double]) < 1e-9, "Failed to exchange coordinates.")
      assert(Math.abs(inputs(i)(numRealInputs).asInstanceOf[Double] - rotatedInputs(i)(2).asInstanceOf[Double]) < 1e-9, "Failed to exchange coordinates.")
      (3 to numRealInputs - 1).foreach { j =>
        assert(Math.abs(inputs(i)(j).asInstanceOf[Double] - rotatedInputs(i)(j).asInstanceOf[Double]) < 1e-9, "Unexpected modification of un-rotated coordinate.")
      }
    }

    // Check that we can undo rotation by applying the transpose
    val unrotatedInputs = FeatureRotator.applyRotation(rotatedInputs, featuresToRotate, U.t)
    inputs.indices.foreach { i =>
      assert(inputs(i)(0) == unrotatedInputs(i)(0), "Failed to leave categorical invariant.")
      (1 to numRealInputs - 1).foreach { j =>
        assert(Math.abs(inputs(i)(j).asInstanceOf[Double] - unrotatedInputs(i)(j).asInstanceOf[Double]) < 1e-9, "Unexpected modification of coordinate after applying transpose.")
      }
    }
  }

  /**
    * Guess the mean should be invariant under rotation.
    */
  @Test
  def testRotatedGTM(): Unit = {
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
    * Linear regression should be invariant under rotation
    */
  @Test
  def testRotatedLinear(): Unit = {
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

    gradient.get.zip(rotatedGradient.get).foreach { case (free, rotated) =>
      val diff = free.zip(rotated).map { case (f, s) => Math.abs(f - s) }.max
      assert(diff < 1.0e-9, "Gradients should be the same")
    }
  }

  /**
    * Ridge regression should not depend on rotation
    */
  @Test
  def testRotatedRidge(): Unit = {
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

  /**
   * Regression trees should not depend on rotation
   */
  @Test
  def testRotatedRegressionTree(): Unit = {
    val learner = RegressionTreeLearner()
    val model = learner.train(data).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1)).getExpected()

    result.zip(rotatedResult).foreach { case (free: Double, rotated: Double) =>
      assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }
  }

  /**
   * Classification trees should not depend on rotation
   */
  @Test
  def testRotatedClassificationTree(): Unit = {
    val classificationData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(2048, 12, noise = 0.1,
        function = Friedman.friedmanSilverman),
      responseBins = Some(2)
    )

    val learner = ClassificationTreeLearner()
    val model = learner.train(classificationData).getModel()
    val result = model.transform(classificationData.map(_._1)).getExpected()

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(classificationData).getModel()
    val rotatedResult = rotatedModel.transform(classificationData.map(_._1)).getExpected()

    result.zip(rotatedResult).foreach { case (free: Any, rotated: Any) =>
      assert(free == rotated, s"${free} and ${rotated} should be the same")
    }
  }

  /**
   * Test that multitask rotation has the proper performance and invariants
   */
  @Test
  def testMultiTaskRotator(): Unit = {
    val data: Vector[(Vector[Double], Double)] = TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)

    // Generate multi-task training data
    val (inputs, doubleLabel) = data.unzip
    val catLabel = inputs.map(x => Friedman.friedmanGrosseSilverman(x) > 15.0)

    // Sparsify the categorical labels
    val sparseCatLabel = catLabel.map(x => if (Random.nextBoolean()) x else null)

    // Train and evaluate rotated models on original and rotated features
    val baseLearner = new MultiTaskTreeLearner()
    val rotator = new MultiTaskFeatureRotator(MultiTaskTreeLearner())

    // Check double labels are the same
    val baseDoubleRes = baseLearner.train(inputs, Seq(doubleLabel, sparseCatLabel)).head.getModel().transform(inputs).getExpected()
    val rotatedDoubleRes = rotator.train(inputs, Seq(doubleLabel, sparseCatLabel)).head.getModel().transform(inputs).getExpected()
    baseDoubleRes.zip(rotatedDoubleRes).foreach { case (br: Double, rr: Double) =>
      assert(Math.abs(br - rr) < 1e-9, "Predicted double label not the same in rotated learner.")
    }

    // Check categorical labels are the same
    val baseTrainingRes = baseLearner.train(inputs, Seq(doubleLabel, sparseCatLabel))
    val baseCatRes = baseLearner.train(inputs, Seq(doubleLabel, sparseCatLabel)).last.getModel().transform(inputs).getExpected()
    val rotatedCatRes = rotator.train(inputs, Seq(doubleLabel, sparseCatLabel)).last.getModel().transform(inputs).getExpected()

    val baseF1 = ClassificationMetrics.f1scores(baseCatRes, catLabel)
    val rotatedF1 = ClassificationMetrics.f1scores(rotatedCatRes, catLabel)
    // rotatedF1 appears to come in be substantially lower than baseF1; this is just a rough sanity check / smoketest.
    assert(Math.abs(baseF1 - rotatedF1) < 0.15)
  }

  /*
   * TODO: investigate performance relative to un-rotated random forest
   */
}
