package io.citrine.lolo.transformers

import breeze.linalg.{det, DenseMatrix}
import io.citrine.lolo.{SeedRandomMixIn, TestUtils}
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import org.junit.Test

/**
  * Created by gregor-robinson on 2020-01-06.
  */
@Test
class FeatureRotatorTest extends SeedRandomMixIn {

  val data: Seq[(Vector[Any], Double)] = {
    val (baseInputs, baseLabels) =
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).unzip
    val binnedInputs = TestUtils.binTrainingInputs(baseInputs, bins = Seq((0, 8)))
    binnedInputs.zip(baseLabels)
  }
  val weights: Vector[Double] = Vector.fill(data.size)(if (rng.nextBoolean()) rng.nextDouble() else 0.0)

  @Test
  def testRandomRotation(): Unit = {
    val inputs = data.map(_._1)
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
    val inputs = data.map(_._1)
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
      rotatedTrainingResult.getLoss() == rotatedTrainingResult.baseTrainingResult.getLoss(),
      "Function getLoss() should pass through to base learner."
    )

    rotatedTrainingResult.getPredictedVsActual().foreach { x =>
      x.zip(data).foreach {
        case (a, b) =>
          assert(a._1 == b._1, "getPredictedVsActual must return the correct training inputs.")
          assert(a._2 == b._2, "getPredictedVsActual must return the correct predicted value.")
          assert(a._3 == b._2, "getPredictedVsActual must return the correct actual value.")
      }
    }
  }

  /**
    * Guess the mean should be invariant under rotation.
    */
  @Test
  def testRotatedGTM(): Unit = {
    val learner = GuessTheMeanLearner()
    val model = learner.train(data, rng = rng).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val rotatedLearner = FeatureRotator(GuessTheMeanLearner())
    val rotatedModel = rotatedLearner.train(data, rng = rng).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1)).getExpected()

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
    val model = learner.train(data, Some(weights), rng).getModel()
    val result = model.transform(data.map(_._1))
    val expected = result.getExpected()
    val gradient = result.getGradient()

    val rotatedLearner = FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data, Some(weights), rng).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1))
    val rotatedExpected = rotatedResult.getExpected()
    val rotatedGradient = rotatedResult.getGradient()

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
    val model = learner.train(data, rng = rng).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data, rng = rng).getModel()
    val rotatedResult = rotatedModel.transform(data.map(_._1)).getExpected()

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
    val model = learner.train(data, rng = rng).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel = rotatedLearner.train(data, rng = rng).getModel().asInstanceOf[RotatedFeatureModel[Double]]
    var rotatedResult = rotatedModel.transform(data.map(_._1)).getExpected()
    result.zip(rotatedResult).foreach {
      case (free: Double, rotated: Double) =>
        assert(Math.abs(free - rotated) < 1.0e-9, s"${free} and ${rotated} should be the same")
    }

    val rotatedData = FeatureRotator.applyRotation(data.map(_._1), rotatedModel.rotatedFeatures, rotatedModel.trans)
    rotatedResult = rotatedModel.transform(rotatedData).getExpected()
    // Check that labels change when we feed in different data.
    assert(
      rotatedResult.zip(result).map { case (a: Double, b: Double) => a - b }.count { x => Math.abs(x) > 1e-9 } > 0,
      "Rotated data passed to rotated model should not map to the same predictions."
    )
    val baseResult = rotatedModel.baseModel.transform(rotatedData).getExpected()
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
    val classificationData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(2048, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng),
      responseBins = Some(2)
    )

    val learner = ClassificationTreeLearner()
    val model = learner.train(classificationData, rng = rng).getModel()
    val result = model.transform(classificationData.map(_._1)).getExpected()

    val rotatedLearner = new FeatureRotator(learner)
    val rotatedModel =
      rotatedLearner.train(classificationData, rng = rng).getModel().asInstanceOf[RotatedFeatureModel[String]]
    var rotatedResult = rotatedModel.transform(classificationData.map(_._1)).getExpected()

    result.zip(rotatedResult).foreach {
      case (free: Any, rotated: Any) =>
        assert(free == rotated, s"${free} and ${rotated} should be the same")
    }

    val rotatedData =
      FeatureRotator.applyRotation(classificationData.map(_._1), rotatedModel.rotatedFeatures, rotatedModel.trans)
    rotatedResult = rotatedModel.transform(rotatedData).getExpected()
    // Check that labels change when we feed in different data.
    assert(
      rotatedResult.zip(result).count { case (a: String, b: String) => a != b } > 0,
      "Rotated data passed to rotated model should not map to the same predictions."
    )
    val baseResult = rotatedModel.baseModel.transform(rotatedData).getExpected()
    // Check that labels are the same as feeding rotated data into base learner.
    assert(
      baseResult.zip(result).count { case (a: String, b: String) => a != b } == 0,
      s"Categorical labels should be identical when we feed rotated inputs into base learner."
    )
  }

  /**
    * Verify that rotated-input multi-task trees have expected properties
    */
  @Test
  def testMultiTaskRotator(): Unit = {
    val data: Vector[(Vector[Double], Double)] =
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)

    // Generate multi-task training data
    val (inputs, doubleLabel) = data.unzip
    val catLabel = inputs.map(x => Friedman.friedmanGrosseSilverman(x) > 15.0)

    // Sparsify the categorical labels
    val sparseCatLabel = catLabel.map(x => if (rng.nextBoolean()) x else null)
    val labels = Vector(doubleLabel, sparseCatLabel).transpose

    // Train and evaluate rotated models on original and rotated features
    val baseLearner = MultiTaskTreeLearner()
    val rotatedLearner = MultiTaskFeatureRotator(MultiTaskTreeLearner())

    val baseTrainingResult = baseLearner.train(inputs.zip(labels), rng = rng)
    val baseDoubleModel = baseTrainingResult.getModels().head
    val baseCatModel = baseTrainingResult.getModels().last
    val rotatedTrainingResult = rotatedLearner.train(inputs.zip(labels))
    val rotatedDoubleModel = rotatedTrainingResult.getModels().head.asInstanceOf[RotatedFeatureModel[Double]]
    val rotatedCatModel = rotatedTrainingResult.getModels().last.asInstanceOf[RotatedFeatureModel[Boolean]]

    // Check double labels are the same
    val baseDoubleResult = baseDoubleModel.transform(inputs).getExpected()
    val rotatedDoubleResult = rotatedDoubleModel.transform(inputs).getExpected()
    baseDoubleResult.zip(rotatedDoubleResult).foreach {
      case (br: Double, rr: Double) =>
        assert(Math.abs(br - rr) < 1e-9, "Predicted double label not the same in rotated learner.")
    }

    // Check categorical labels are close
    val baseCatResult = baseCatModel.transform(inputs).getExpected()
    val rotatedCatResult = rotatedCatModel.transform(inputs).getExpected()
    val baseF1 = ClassificationMetrics.f1scores(baseCatResult, catLabel)
    val rotatedF1 = ClassificationMetrics.f1scores(rotatedCatResult, catLabel)
    // rotatedF1 appears to come in substantially lower than baseF1; this is just a rough sanity check / smoketest.
    assert(Math.abs(baseF1 - rotatedF1) < 0.15, s"baseF1 ${baseF1} and rotatedF1 ${rotatedF1} are too dissimilar.")

    // Check that rotation features and rotation matrices are the same for both tasks
    val U = rotatedDoubleModel.trans
    val rotatedFeatures = rotatedDoubleModel.rotatedFeatures
    assert(U == rotatedCatModel.trans)
    assert(rotatedFeatures == rotatedCatModel.rotatedFeatures)

    val rotatedInputs = FeatureRotator.applyRotation(data.map(_._1), rotatedFeatures, U)
    assert(
      rotatedDoubleModel.baseModel.transform(rotatedInputs).getExpected().zip(rotatedDoubleResult).count {
        case (a: Double, b: Double) => Math.abs(a - b) > 1e-9
      } == 0,
      "Rotated data passed to base model should map to the same predictions."
    )
    assert(
      rotatedDoubleModel.transform(rotatedInputs).getExpected().zip(baseDoubleResult).count {
        case (a: Double, b: Double) => Math.abs(a - b) > 1e-9
      } > 0,
      "Rotated data passed to rotated model should map to different predictions."
    )

    val doublyRotatedF1 =
      ClassificationMetrics.f1scores(rotatedCatModel.transform(rotatedInputs).getExpected(), catLabel)
    val rotatedF1Base = ClassificationMetrics.f1scores(baseCatModel.transform(rotatedInputs).getExpected(), catLabel)

    // Check that categorical labels change when we feed in different data.
    assert(
      rotatedCatModel
        .transform(rotatedInputs)
        .getExpected()
        .zip(baseCatModel.transform(rotatedInputs).getExpected())
        .count {
          case (a: Boolean, b: Boolean) => a != b
        } > 0,
      s"Categorical labels should substantially change when we feed in different inputs."
    )
    // Check that categorical labels are the same as feeding rotated data into base learner.
    assert(
      rotatedCatModel
        .transform(inputs)
        .getExpected()
        .zip(rotatedCatModel.baseModel.transform(rotatedInputs).getExpected)
        .count {
          case (a: Boolean, b: Boolean) => a != b
        } == 0,
      s"Categorical labels should be identical when we feed rotated inputs into base learner."
    )
  }
}
