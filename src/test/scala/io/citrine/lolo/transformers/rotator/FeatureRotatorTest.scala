package io.citrine.lolo.transformers.rotator

import breeze.linalg.{det, DenseMatrix}
import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import org.junit.Test

/**
  * Created by gregor-robinson on 2020-01-06.
  */
@Test
class FeatureRotatorTest extends SeedRandomMixIn {

  val data: Seq[TrainingRow[Double]] = DataGenerator
    .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
    .withBinnedInputs(bins = Seq((0, 8)))
    .data
  val weightedData: Seq[TrainingRow[Double]] = data.map { row =>
    row.withWeight(if (rng.nextBoolean()) rng.nextDouble() else 0.0)
  }

  @Test
  def testRandomRotation(): Unit = {
    val inputs = data.map(_.inputs)
    val numRealInputs = inputs.head.length - 1
    for (i <- 1 to 10) {
      val U = FeatureRotator.getRandomRotation(numRealInputs)
      assert(U.rows == numRealInputs)
      assert(U.cols == numRealInputs)

      // Check that the matrix is unitary.
      (U.t * U - DenseMatrix.eye[Double](numRealInputs)).toArray.foreach { x =>
        assert(Math.abs(x) < 1e-9)
      }

      // Check that the determinant is one.
      val d = det(U)
      assert(Math.abs(d - 1.0) < 1e-9, s"Determinant of U should be 1 but is ${d}")
    }
  }

  @Test
  def testApplyRotation(): Unit = {
    val inputs = data.map(_.inputs)
    val numRealInputs = inputs.head.length - 1
    val featuresToRotate = (1 to numRealInputs).asInstanceOf[IndexedSeq[Int]]

    // Create rotation matrix that exchanges first, second, and last real inputs
    val U = DenseMatrix.eye[Double](dim = numRealInputs)
    U(0, 0) = 0.0
    U(1, 1) = 0.0
    U(numRealInputs - 1, numRealInputs - 1) = 0.0
    U(0, 1) = 1.0
    U(1, numRealInputs - 1) = 1.0
    U(numRealInputs - 1, 0) = 1.0

    val rotatedInputs = FeatureRotator.applyRotation(inputs, featuresToRotate, U)
    inputs.indices.foreach { i =>
      assert(inputs(i)(0) == rotatedInputs(i)(0), "Failed to leave categorical invariant.")
      assert(
        Math.abs(inputs(i)(1).asInstanceOf[Double] - rotatedInputs(i)(numRealInputs).asInstanceOf[Double]) < 1e-9,
        "Failed to exchange coordinates."
      )
      assert(
        Math.abs(inputs(i)(2).asInstanceOf[Double] - rotatedInputs(i)(1).asInstanceOf[Double]) < 1e-9,
        "Failed to exchange coordinates."
      )
      assert(
        Math.abs(inputs(i)(numRealInputs).asInstanceOf[Double] - rotatedInputs(i)(2).asInstanceOf[Double]) < 1e-9,
        "Failed to exchange coordinates."
      )
      (3 to numRealInputs - 1).foreach { j =>
        assert(
          Math.abs(inputs(i)(j).asInstanceOf[Double] - rotatedInputs(i)(j).asInstanceOf[Double]) < 1e-9,
          "Unexpected modification of un-rotated coordinate."
        )
      }
    }

    // Check that we can undo rotation by applying the transpose
    val unrotatedInputs = FeatureRotator.applyRotation(rotatedInputs, featuresToRotate, U.t)
    inputs.indices.foreach { i =>
      assert(inputs(i)(0) == unrotatedInputs(i)(0), "Failed to leave categorical invariant.")
      (1 to numRealInputs - 1).foreach { j =>
        assert(
          Math.abs(inputs(i)(j).asInstanceOf[Double] - unrotatedInputs(i)(j).asInstanceOf[Double]) < 1e-9,
          "Unexpected modification of coordinate after applying transpose."
        )
      }
    }
  }

  /**
    * Functions getLoss and getPredictedVsActual should pass through to base learner.
    */
  @Test
  def testPassthroughFunctions(): Unit = {
    val rotatedTrainingResult = FeatureRotator(RegressionTreeLearner()).train(data, rng = rng)

    assert(
      rotatedTrainingResult.loss == rotatedTrainingResult.baseTrainingResult.loss,
      "Function getLoss() should pass through to base learner."
    )

    rotatedTrainingResult.predictedVsActual.foreach { x =>
      x.zip(data).foreach {
        case (a, b) =>
          assert(a._1 == b.inputs, "getPredictedVsActual must return the correct training inputs.")
          assert(a._2 == b.label, "getPredictedVsActual must return the correct predicted value.")
          assert(a._3 == b.label, "getPredictedVsActual must return the correct actual value.")
      }
    }
  }

  /**
    * Guess the mean should be invariant under rotation.
    */
  @Test
  def testRotatedGTM(): Unit = {
    val learner = GuessTheMeanLearner()
    val model = learner.train(data, rng = rng).model
    val result = model.transform(data.map(_.inputs)).expected

    val rotatedLearner = FeatureRotator(GuessTheMeanLearner())
    val rotatedModel = rotatedLearner.train(data, rng = rng).model
    val rotatedResult = rotatedModel.transform(data.map(_.inputs)).expected

    result.zip(rotatedResult).foreach {
      case (free: Double, rotated: Double) =>
        assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }
  }

  /**
    * Linear regression should be invariant under rotation
    */
  @Test
  def testRotatedLinear(): Unit = {
    val learner = LinearRegressionLearner()
    val model = learner.train(weightedData, rng).model
    val result = model.transform(data.map(_.inputs))
    val expected = result.expected
    val gradient = result.gradient

    val rotatedLearner = FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(weightedData, rng).model
    val rotatedResult = rotatedModel.transform(data.map(_.inputs))
    val rotatedExpected = rotatedResult.expected
    val rotatedGradient = rotatedResult.gradient

    expected.zip(rotatedExpected).foreach {
      case (free: Double, rotated: Double) =>
        assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }

    gradient.get.zip(rotatedGradient.get).foreach {
      case (free, rotated) =>
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
    val model = learner.train(data, rng = rng).model
    val result = model.transform(data.map(_.inputs)).expected

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data, rng = rng).model
    val rotatedResult = rotatedModel.transform(data.map(_.inputs)).expected

    result.zip(rotatedResult).foreach {
      case (free: Double, rotated: Double) =>
        assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }
  }

  /**
    * Verify expected properties of rotated-input regression trees
    */
  @Test
  def testRotatedRegressionTree(): Unit = {
    val learner = RegressionTreeLearner()
    val model = learner.train(data, rng = rng).model
    val result = model.transform(data.map(_.inputs)).expected

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data, rng = rng).model.asInstanceOf[RotatedFeatureModel[Double]]
    var rotatedResult = rotatedModel.transform(data.map(_.inputs)).expected
    result.zip(rotatedResult).foreach {
      case (free: Double, rotated: Double) =>
        assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }

    val rotatedData = FeatureRotator.applyRotation(data.map(_.inputs), rotatedModel.rotatedFeatures, rotatedModel.trans)
    rotatedResult = rotatedModel.transform(rotatedData).expected
    // Check that labels change when we feed in different data.
    assert(
      rotatedResult.zip(result).map { case (a: Double, b: Double) => a - b }.count { x => Math.abs(x) > 1e-9 } > 0,
      "Rotated data passed to rotated model should not map to the same predictions."
    )
    val baseResult = rotatedModel.baseModel.transform(rotatedData).expected
    // Check that labels are the same as feeding rotated data into base learner.
    assert(
      baseResult.zip(result).map { case (a: Double, b: Double) => a - b }.count { x => Math.abs(x) > 1e-9 } == 0,
      "Rotated data passed to base model should map to the same predictions."
    )
  }

  /**
    * Verify expected properties of rotated-input classification trees
    */
  @Test
  def testRotatedClassificationTree(): Unit = {
    val classificationData = DataGenerator
      .generate(2048, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedLabels(bins = 2)
      .data

    val learner = ClassificationTreeLearner()
    val model = learner.train(classificationData, rng = rng).model
    val result = model.transform(classificationData.map(_.inputs)).expected

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel =
      rotatedLearner.train(classificationData, rng = rng).model.asInstanceOf[RotatedFeatureModel[String]]
    var rotatedResult = rotatedModel.transform(classificationData.map(_.inputs)).expected

    result.zip(rotatedResult).foreach {
      case (free: Any, rotated: Any) =>
        assert(free == rotated, s"${free} and ${rotated} should be the same")
    }

    val rotatedData =
      FeatureRotator.applyRotation(classificationData.map(_.inputs), rotatedModel.rotatedFeatures, rotatedModel.trans)
    rotatedResult = rotatedModel.transform(rotatedData).expected
    // Check that labels change when we feed in different data.
    assert(
      rotatedResult.zip(result).count { case (a: String, b: String) => a != b } > 0,
      "Rotated data passed to rotated model should not map to the same predictions."
    )
    val baseResult = rotatedModel.baseModel.transform(rotatedData).expected
    // Check that labels are the same as feeding rotated data into base learner.
    assert(
      baseResult.zip(result).count { case (a: String, b: String) => a != b } == 0,
      s"Categorical labels should be identical when we feed rotated inputs into base learner."
    )
  }
}
