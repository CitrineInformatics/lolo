package io.citrine.lolo.trees

import io.citrine.lolo.TestUtils
import org.junit.Test

/**
  * Created by maxhutch on 12/2/16.
  */
@Test
class ClassificationTreeTest {

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def longerTest(): Unit = {
    val csv = TestUtils.readCsv("class_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last))
    val DTLearner = new ClassificationTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.getModel()
    (0 until N).map(i => DTLearner.train(trainingData))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    trainingData.foreach { case (x, y) =>
      assert(y == DT.transform(Seq(x)).getExpected().head)
    }

    /* The first feature should be the most important */
    val importances = DTMeta.getFeatureImportance()
    println(importances.toList)
    assert(importances(0) == importances.max)
  }

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def testCategorical(): Unit = {
    val csv = TestUtils.readCsv("class_example_with_cat.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last))
    val DTLearner = new ClassificationTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DT = DTLearner.train(trainingData).getModel()
    (0 until N).map(i => DTLearner.train(trainingData))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    trainingData.foreach { case (x, y) =>
      assert(y == DT.transform(Seq(x)).getExpected().head)
    }
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
    new ClassificationTreeTest().longerTest()
    // new ClassificationTreeTest().testCategorical()
  }
}
