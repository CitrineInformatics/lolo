package io.citrine.lolo.trees.regression

import java.io.{File, FileOutputStream, ObjectOutputStream}

import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import org.junit.Test
import org.scalatest.Assertions._

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class RegressionTreeTest {

  /**
    * Trivial models with no splits should have finite feature importance.
    */
  @Test
  def testFeatureImportanceNaN(): Unit = {
    val X = Vector.fill(100) {
      val input = Vector.fill(10)(1.0)
      (input, 2.0)
    }

    val DTLearner = new RegressionTreeLearner()
    val DTMeta = DTLearner.train(X)
    val DT = DTMeta.getModel()
    assert(DTMeta.getFeatureImportance()
      .get.forall(v => !v.isNaN))
  }

  /**
    * Test a simple tree with only real inputs
    */
  @Test
  def testSimpleTree(): Unit = {
    val csv = TestUtils.readCsv("double_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
    val DTLearner = new RegressionTreeLearner()
    val DT = DTLearner.train(trainingData).getModel()

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_._1))
    trainingData.zip(output.getExpected()).foreach { case ((x, a), p) =>
      assert(Math.abs(a - p) < 1.0e-9)
    }
    assert(output.getGradient().isEmpty)
    output.getDepth().foreach(d => assert(d > 3 && d < 9, s"Depth is ${d}"))
  }

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def longerTest(): Unit = {
    val trainingData =TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val DTLearner = new RegressionTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.getModel()
    (0 until N).map(i => DTLearner.train(trainingData).getModel())
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_._1))
    trainingData.zip(output.getExpected()).foreach { case ((x, a), p) =>
      assert(Math.abs(a - p) < 1.0e-9)
    }
    assert(output.getGradient().isEmpty)
    output.getDepth().foreach(d => assert(d > 4 && d < 20, s"Depth is ${d}"))

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance().get
    assert(importances(1) == importances.max)

    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Train with a categorical variable
    */
  @Test
  def testCategorical(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    ).asInstanceOf[Seq[(Vector[Any], Double)]]

    val DTLearner = new RegressionTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.getModel()
    (0 until N).map(i => DTLearner.train(trainingData).getModel())
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_._1))
    trainingData.zip(output.getExpected()).foreach { case ((x, a), p) =>
      assert(Math.abs(a - p) < 1.0e-9)
    }
    assert(output.getGradient().isEmpty)
    output.getDepth().foreach(d => assert(d > 4 && d < 21, s"Depth is ${d}"))

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance().get
    assert(importances(1) == importances.max)
    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Train with linear leaves.  This case is under-constriained, so we should hit the results exactly
    */
  @Test
  def testLinearLeaves(): Unit = {
    val trainingData = TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)

    val linearLearner = new LinearRegressionLearner().setHyper("regParam", 0.0)
    val DTLearner = new RegressionTreeLearner(leafLearner = Some(linearLearner)).setHyper("minLeafInstances", 2)
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.getModel()

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_._1))
    trainingData.zip(output.getExpected()).foreach { case ((x, a), p) =>
      assert(Math.abs(a - p) < 1.0e-9)
    }
    assert(output.getGradient().isDefined)
    output.getDepth().foreach(d => assert(d > 4 && d < 18, s"Depth is ${d}"))

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance().get
    assert(importances(1) == importances.max)

    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }
}

/** Companion driver */
object RegressionTreeTest {
  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    new RegressionTreeTest().testSimpleTree()
    new RegressionTreeTest().longerTest()
    new RegressionTreeTest().testCategorical()
    new RegressionTreeTest().testLinearLeaves()
  }
}
