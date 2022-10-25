package io.citrine.lolo.trees.classification

import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TrainingRow}
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
