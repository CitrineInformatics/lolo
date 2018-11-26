package io.citrine.lolo.transformers

import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.{GuessTheMeanLearner, LinearRegressionLearner}
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 2/19/17.
  */
@Test
class StandardizerTest {

  val data: Vector[(Vector[Double], Double)] = TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)
  val weights: Vector[Double] = Vector.fill(data.size)(if (Random.nextBoolean()) Random.nextDouble() else 0.0)

  // Creating another dataset which has 1 feature that has 0 variance.
  val dataWithConstant: Vector[(Vector[Double], Double)] = data.map(d => (0.0 +: d._1, d._2))

  @Test
  def testStandardMeanAndVariance(): Unit = {
    val inputs = data.map(_._1)
    val inputTransforms = Standardizer.getMultiStandardization(inputs)

    val standardInputs = Standardizer.applyStandardization(inputs, inputTransforms)
    standardInputs.head.indices.foreach { i =>
      val values = standardInputs.map(_ (i).asInstanceOf[Double])
      val mean = values.sum / values.size
      assert(mean < 1.0e-9, s"Standard input ${i} has non-zero mean ${mean}")

      val variance = values.map(Math.pow(_, 2)).sum / values.size
      assert(Math.abs(variance - 1.0) < 1.0e-9, s"Standard input ${i} has non-unit variance ${variance}")
    }

    val outputs = data.map(_._2.asInstanceOf[Double])
    val outputTransform = Standardizer.getStandardization(outputs)
    val standardOutputs = Standardizer.applyStandardization(outputs, Some(outputTransform)).asInstanceOf[Seq[Double]]

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
    val model = learner.train(data).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val standardLearner = Standardizer(GuessTheMeanLearner())
    val standardModel = standardLearner.train(data).getModel()
    val standardResult = standardModel.transform(data.map(_._1)).getExpected()

    result.zip(standardResult).foreach { case (free: Double, standard: Double) =>
      assert(Math.abs(free - standard) < 1.0e-9, s"${free} and ${standard} should be the same")
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

    val standardLearner = Standardizer(learner)
    val standardModel = standardLearner.train(data, Some(weights)).getModel()
    val standardResult = standardModel.transform(data.map(_._1))
    val standardExpected = standardResult.getExpected()
    val standardGradient = standardResult.getGradient()

    expected.zip(standardExpected).foreach { case (free: Double, standard: Double) =>
      assert(Math.abs(free - standard) < 1.0e-9, s"${free} and ${standard} should be the same")
    }

    gradient.get.zip(standardGradient.get).foreach { case (free, standard) =>
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
    val model = learner.train(dataWithConstant, Some(weights)).getModel()
    val result = model.transform(dataWithConstant.map(_._1))
    val expected = result.getExpected()
    val gradient = result.getGradient()

    val standardLearner = Standardizer(learner)
    val standardModel = standardLearner.train(dataWithConstant, Some(weights)).getModel()
    val standardResult = standardModel.transform(dataWithConstant.map(_._1))
    val standardExpected = standardResult.getExpected()
    val standardGradient = standardResult.getGradient()


    gradient.get.toList.flatten.zip(standardGradient.get.toList.flatten).foreach { case (free: Double, standard: Double) =>
      assert(Math.abs(free - standard) < 1.0e-9, s"Failed test for gradient. ${free} and ${standard} gradients should be the same")
    }

    // The gradient wrt the first constant feature is ill-defined without regularization
    gradient.get.zip(standardGradient.get).foreach { case (free, standard) =>
      val diff = free.zip(standard).map { case (f, s) => Math.abs(f - s) }.max
      assert(diff < 1.0e-9, s"Gradients should be the same. The diff is $diff")
    }

    // This test fails ~30% of the time when nRows=30, but its not clear why.
    expected.zip(standardExpected).foreach { case (free: Double, standard: Double) =>
      assert(Math.abs(free - standard) < 1.0e-9, s"Failed test for expected. ${free} and ${standard} should be the same")
    }


  }


  /**
    * Ridge regression should depend on standardization
    */
  @Test
  def testStandardRidge(): Unit = {
    val learner = LinearRegressionLearner(regParam = Some(1.0))
    val model = learner.train(data).getModel()
    val result = model.transform(data.map(_._1)).getExpected()

    val standardLearner = new Standardizer(learner)
    val standardModel = standardLearner.train(data).getModel()
    val standardResult = standardModel.transform(data.map(_._1)).getExpected()

    result.zip(standardResult).foreach { case (free: Double, standard: Double) =>
      assert(Math.abs(free - standard) > 1.0e-9, s"${free} and ${standard} should NOT be the same")
    }
  }

  @Test
  def testStandardClassification(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(2048, 12, noise = 0.1,
        function = Friedman.friedmanSilverman),
      responseBins = Some(2)
    )

    val learner = ClassificationTreeLearner()
    val model = learner.train(trainingData).getModel()
    val result = model.transform(trainingData.map(_._1)).getExpected()

    val standardLearner = Standardizer(learner)
    val standardModel = standardLearner.train(trainingData).getModel()
    val standardResult = standardModel.transform(trainingData.map(_._1)).getExpected()
    result.zip(standardResult).foreach { case (free: String, standard: String) =>
      assert(free == standard, s"Standard classification tree should be the same")
    }
  }

  /**
    * Test that multitask sandardization has the proper performance and invariants
    */
  @Test
  def testMultiTaskStandardizer(): Unit = {
    // Generate multi-task training data
    val (inputs, doubleLabel) = data.unzip
    val catLabel = inputs.map(x => Friedman.friedmanGrosseSilverman(x) > 15.0)

    // Sparsify the categorical labels
    val sparseCatLabel = catLabel.map(x => if (Random.nextBoolean()) x else null)

    // Screw up the scale of the double labels
    val scale = 10000.0
    val rescaledLabel = doubleLabel.map(_ * scale)

    // Train and evaluate standard models on original and rescaled labels
    val standardizer = new MultiTaskStandardizer(MultiTaskTreeLearner())
    val baseRes = standardizer.train(inputs, Seq(doubleLabel, sparseCatLabel)).last.getModel().transform(inputs).getExpected()
    val standardRes = standardizer.train(inputs, Seq(rescaledLabel, sparseCatLabel)).last.getModel().transform(inputs).getExpected()
    // Train and evaluate unstandardized model on rescaled labels

    // Compute metrics for each of the models
    val baseF1 = ClassificationMetrics.f1scores(baseRes, catLabel)
    val standardF1 = ClassificationMetrics.f1scores(standardRes, catLabel)

    // Assert some things
    assert(Math.abs(baseF1 - standardF1) < 1.0e-6, s"Expected training to be invariant the scale of the labels")
  }
}
