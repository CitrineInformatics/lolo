package io.citrine.lolo

import io.citrine.lolo.trees.SimpleRegressionTreeLearner
import org.junit.Test

/**
  * Created by maxhutch on 11/28/16.
  */
@Test
class SimpleRegressionTreeTest {

  @Test
  def testSimpleTree(): Unit = {
    val csv = TestUtils.readCsv("double_example.csv")
    val trainingData = csv.map(vec => (vec.init.asInstanceOf[Vector[Double]], vec.last.asInstanceOf[Double]))
    val DTLearner = new SimpleRegressionTreeLearner()
    val DT = DTLearner.train(trainingData)
    trainingData.foreach { case (x, y) =>
        assert(y == DT.predict(x))
    }
  }

  @Test
  def longerTest(): Unit = {
    val csv = TestUtils.readCsv("large_example.csv")
    val trainingData = csv.map(vec => (vec.init.asInstanceOf[Vector[Double]], vec.last.asInstanceOf[Double]))
    val DTLearner = new SimpleRegressionTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DT = DTLearner.train(trainingData)
    (0 until N).map(i => DTLearner.train(trainingData))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration/N} s")

    /* We should be able to memorize the inputs */
    trainingData.foreach { case (x, y) =>
        assert(y == DT.predict(x))
    }

    /* The first feature should be the most important */
    val importances = DT.getFeatureImportance()
    println(importances.toList)
    assert(importances(0) == importances.max)
  }

}

object SimpleRegressionTreeTest {
  def main(argv: Array[String]): Unit = {
    // new SimpleRegressionTreeTest().testSimpleTree()
    new SimpleRegressionTreeTest().longerTest()
  }
}
