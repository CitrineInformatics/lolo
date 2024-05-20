package io.citrine.lolo.trees.classification

import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn}
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test
import org.scalatest.Assertions._

@Test
class ClassificationTreeTest extends SeedRandomMixIn {

  /**
    * Trivial models with no splits should have finite feature importance.
    */
  @Test
  def testFeatureImportanceNaN(): Unit = {
    val X = Vector.fill(100) {
      val input = Vector.fill(10)(1.0)
      TrainingRow(input, 2.0)
    }

    val DTLearner = ClassificationTreeLearner()
    val DTMeta = DTLearner.train(X)
    assert(DTMeta.featureImportance.get.forall(v => !v.isNaN))
  }

  @Test
  def testBinary(): Unit = {
    val trainingData = DataGenerator
      .generate(2048, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedLabels(bins = 2)
      .data

    val DTLearner = ClassificationTreeLearner()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.model

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(row.label == p, s"${row.label} != $p for ${row.inputs}")
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 0))

    /* The first features should be the most important */
    val importances = DTMeta.featureImportance.get
    assert(importances.slice(0, 5).min > importances.slice(5, importances.size).max)
  }

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def longerTest(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.05, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedLabels(bins = 16)
      .data

    val DTLearner = ClassificationTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.model
    (0 until N).map(i => DTLearner.train(trainingData))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) => assert(row.label == p, s"${row.label} != $p for ${row.inputs}")
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 4 && d < 17, s"Depth is ${d}"))

    /* The first feature should be the most important */
    val importances = DTMeta.featureImportance.get
    assert(importances.slice(0, 5).min > importances.slice(5, importances.size).max)
  }

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def testCategorical(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .withBinnedLabels(bins = 16)
      .data

    val DTLearner = ClassificationTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DT = DTLearner.train(trainingData).model
    (0 until N).map(i => DTLearner.train(trainingData))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(row.label == p)
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 3 && d < 18, s"Depth is ${d}"))
  }

  @Test
  def testShapley(): Unit = {

    val shapTrainingData = Seq(
      TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 0.0), "1"),
      TrainingRow(Vector(1.0, 0.0, 0.0, 0.0, 0.0), "1"),
      TrainingRow(Vector(0.0, 1.0, 0.0, 0.0, 0.0), "1"),
      TrainingRow(Vector(0.0, 0.0, 1.0, 0.0, 0.0), "0"),
      TrainingRow(Vector(0.0, 0.0, 0.0, 1.0, 0.0), "0"),
      TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 1.0), "0")
    )

    // Create learner and train
    val DTLearner = ClassificationTreeLearner()
    val DTMeta = DTLearner.train(shapTrainingData)
    val DT = DTMeta.model
    // baseValue is the average of the training labels
    val baseValue = 0.5

    // Assert we obtain Shapley values
    val shapley1 = DT.shapley(Vector.fill[Any](5)(1.0))
    assert(shapley1.isDefined, "Shapley values should be defined")
    // Shapley values should sum to the difference between the prediction and the base value
    val shapley1Sum = shapley1.get.valuesIterator.sum
    assert(shapley1Sum + baseValue - 1 < 1e-6, s"Shapley sum is $shapley1Sum")
    // Double check that the prediction is as expected
    val pred1 = DT.transform(Seq(Vector.fill[Any](5)(0.0))).expected.head
    assert(pred1 == "1", "Prediction should be 1")

    // Same as above, but with all zeros
    val shapley0 = DT.shapley1(Vector.fill[Any](5)(0.0))
    assert(shapley0.isDefined, "Shapley values should be defined")

    val shapley0Sum = shapley0.get.valuesIterator.sum
    assert(shapley0Sum + baseValue - 0 < 1e-6, s"Shapley sum is $shapley0Sum")

    val pred0 = DT.transform(Seq(Vector.fill[Any](5)(1.0))).expected.head
    assert(pred0 == "0", "Prediction should be 0")
  }
}

/** Companion driver */
object ClassificationTreeTest {

  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    new ClassificationTreeTest().testBinary()
    new ClassificationTreeTest().longerTest()
    new ClassificationTreeTest().testCategorical()
  }
}
