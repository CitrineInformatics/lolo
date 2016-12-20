package io.citrine.lolo.trees

import java.io.{File, FileOutputStream, ObjectOutputStream}

import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.LinearRegressionLearner
import org.junit.Test

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class RegressionTreeTest {

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
  }

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def longerTest(): Unit = {
    val csv = TestUtils.readCsv("large_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
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
    assert(output.getDepth().forall(_ > 0))
    println(output.getDepth())

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance()
    println(importances.toList)
    assert(importances(0) == importances.max)

    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Train with a categorical variable
    */
  @Test
  def testCategorical(): Unit = {
    val csv = TestUtils.readCsv("large_example_with_cat.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
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

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance()
    assert(importances(0) == importances.max)
        val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Train with linear leaves.  This case is under-constriained, so we should hit the results exactly
    */
  @Test
  def testLinearLeaves(): Unit = {
    val csv = TestUtils.readCsv("large_example_with_cat.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
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

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance()
    assert(importances(0) == importances.max)

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
    // new RegressionTreeTest().testSimpleTree()
    new RegressionTreeTest().longerTest()
    // new RegressionTreeTest().testCategorical()
    // new RegressionTreeTest().testLinearLeaves()
  }
}
