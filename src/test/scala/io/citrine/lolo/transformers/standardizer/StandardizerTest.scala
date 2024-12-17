package io.citrine.lolo.transformers.standardizer

import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import org.junit.Test

@Test
class StandardizerTest extends SeedRandomMixIn {

  val data: Seq[TrainingRow[Double]] =
    DataGenerator.generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
  val weights: Vector[Double] = Vector.fill(data.size)(if (rng.nextBoolean()) rng.nextDouble() else 0.0)
  // Creating another dataset which has 1 feature that has 0 variance.
  val dataWithConstant: Seq[TrainingRow[Double]] = data.map(_.mapInputs(x => 0 +: x))

  val weightedData: Seq[TrainingRow[Double]] = data.zip(weights).map { case (row, w) => row.withWeight(w) }
  val weightedDataWithConstant: Seq[TrainingRow[Double]] =
    dataWithConstant.zip(weights).map { case (row, w) => row.withWeight(w) }

  @Test
  def testStandardMeanAndVariance(): Unit = {
    val inputs = data.map(_.inputs)
    val inputTransforms = Standardization.buildMulti(inputs)

    val standardInputs = inputs.map(input => Standardization.applyMulti(input, inputTransforms))
    standardInputs.head.indices.foreach { i =>
      val values = standardInputs.map(_(i).asInstanceOf[Double])
      val mean = values.sum / values.size
      assert(mean < 1.0e-9, s"Standard input ${i} has non-zero mean ${mean}")

      val variance = values.map(Math.pow(_, 2)).sum / values.size
      assert(Math.abs(variance - 1.0) < 1.0e-9, s"Standard input ${i} has non-unit variance ${variance}")
    }

    val outputs = data.map(_.label)
    val outputTransform = Standardization.build(outputs)
    val standardOutputs = outputs.map(outputTransform.apply)

    val mean = standardOutputs.sum / standardOutputs.size
    assert(mean < 1.0e-9, s"Standard output has non-zero mean ${mean}")

    val variance = standardOutputs.map(Math.pow(_, 2)).sum / standardOutputs.size
    assert(Math.abs(variance - 1.0) < 1.0e-9, s"Standard output has non-unit variance ${variance}")
  }

  /**
    * Guess the mean should be invariant under standardization
    */
  @Test
  def testStandardGTM(): Unit = {
    val learner = GuessTheMeanLearner()
    val model = learner.train(data, rng = rng).model
    val result = model.transform(data.map(_.inputs)).expected

    val standardLearner = RegressionStandardizer(GuessTheMeanLearner())
    val standardModel = standardLearner.train(data, rng = rng).model
    val standardResult = standardModel.transform(data.map(_.inputs)).expected

    result.zip(standardResult).foreach {
      case (free: Double, standard: Double) =>
        assert(Math.abs(free - standard) < 1.0e-9, s"${free} and ${standard} should be the same")
    }
  }

  /**
    * Linear regression should be invariant under standardization
    */
  @Test
  def testStandardLinear(): Unit = {
    val learner = LinearRegressionLearner()
    val model = learner.train(weightedData).model
    val result = model.transform(data.map(_.inputs))
    val expected = result.expected
    val gradient = result.gradient

    val standardLearner = RegressionStandardizer(learner)
    val standardModel = standardLearner.train(weightedData).model
    val standardResult = standardModel.transform(data.map(_.inputs))
    val standardExpected = standardResult.expected
    val standardGradient = standardResult.gradient

    expected.zip(standardExpected).foreach {
      case (free: Double, standard: Double) =>
        assert(Math.abs(free - standard) < 1.0e-9, s"${free} and ${standard} should be the same")
    }

    gradient.get.zip(standardGradient.get).foreach {
      case (free, standard) =>
        val diff = free.zip(standard).map { case (f, s) => Math.abs(f - s) }.max
        assert(diff < 1.0e-9, "Gradients should be the same")
    }
  }

  /**
    * When the variance of a particular feature is 0
    *
    * @author Astha Garg
    */
  @Test
  def testStandardWithConstantFeature(): Unit = {
    val learner = LinearRegressionLearner()
    val model = learner.train(weightedDataWithConstant).model
    val result = model.transform(dataWithConstant.map(_.inputs))
    val expected = result.expected
    val gradient = result.gradient

    val standardLearner = RegressionStandardizer(learner)
    val standardModel = standardLearner.train(weightedDataWithConstant).model
    val standardResult = standardModel.transform(dataWithConstant.map(_.inputs))
    val standardExpected = standardResult.expected
    val standardGradient = standardResult.gradient

    gradient.get.toList.flatten.zip(standardGradient.get.toList.flatten).foreach {
      case (free: Double, standard: Double) =>
        assert(
          Math.abs(free - standard) < 1.0e-9,
          s"Failed test for gradient. ${free} and ${standard} gradients should be the same"
        )
    }

    // The gradient wrt the first constant feature is ill-defined without regularization
    gradient.get.zip(standardGradient.get).foreach {
      case (free, standard) =>
        val diff = free.zip(standard).map { case (f, s) => Math.abs(f - s) }.max
        assert(diff < 1.0e-9, s"Gradients should be the same. The diff is $diff")
    }

    // This test fails ~30% of the time when nRows=30, but its not clear why.
    expected.zip(standardExpected).foreach {
      case (free: Double, standard: Double) =>
        assert(
          Math.abs(free - standard) < 1.0e-9,
          s"Failed test for expected. ${free} and ${standard} should be the same"
        )
    }
  }

  /**
    * Ridge regression should depend on standardization
    */
  @Test
  def testStandardRidge(): Unit = {
    val learner = LinearRegressionLearner(regParam = Some(1.0))
    val model = learner.train(data).model
    val result = model.transform(data.map(_.inputs)).expected

    val standardLearner = RegressionStandardizer(learner)
    val standardModel = standardLearner.train(data).model
    val standardResult = standardModel.transform(data.map(_.inputs)).expected

    result.zip(standardResult).foreach {
      case (free: Double, standard: Double) =>
        assert(Math.abs(free - standard) > 1.0e-9, s"${free} and ${standard} should NOT be the same")
    }
  }

  @Test
  def testStandardClassification(): Unit = {
    val trainingData = DataGenerator
      .generate(2048, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedLabels(bins = 2)
      .data

    val learner = ClassificationTreeLearner()
    val model = learner.train(trainingData).model
    val result = model.transform(trainingData.map(_.inputs)).expected

    val standardLearner = ClassificationStandardizer(learner)
    val standardModel = standardLearner.train(trainingData).model
    val standardResult = standardModel.transform(trainingData.map(_.inputs)).expected
    result.zip(standardResult).foreach {
      case (free: String, standard: String) =>
        assert(free == standard, s"Standard classification tree should be the same")
    }
  }

  /**
    * Shapley values should be calculated on the standardized tree
    */
  @Test
  def testStandardTreeShapley(): Unit = {
    val learner = RegressionTreeLearner()
    val standardLearner = RegressionStandardizer(learner)
    val standardModel = standardLearner.train(data).model
    val nonstandardModel = learner.train(data).model

    data.foreach {
      case row =>
        val shapleyStandardOpt = standardModel.shapley(row.inputs)
        val shapleyStandard = shapleyStandardOpt match {
          case Some(matrix) => matrix.toDenseVector.toScalaVector
          case None         => Vector.empty[Double]
        }
        val standardResult = standardModel.transform(Seq(row.inputs)).expected.head
        val averageStandard = standardResult - shapleyStandard.sum

        val shapleyNonstandardOpt = nonstandardModel.shapley(row.inputs)

        val shapleyNonstandard = shapleyNonstandardOpt match {
          case Some(matrix) => matrix.toDenseVector.toScalaVector
          case None         => Vector.empty[Double]
        }
        val nonstandardResult = nonstandardModel.transform(Seq(row.inputs)).expected.head
        val averageNonStandard = nonstandardResult - shapleyNonstandard.sum

        assert(
          Math.abs(averageStandard - averageNonStandard) < 1.0e-5,
          s"Average shapley values should be the same"
        )

        assert(shapleyStandardOpt.isDefined, "Shapley should be defined")
    }
  }
}
