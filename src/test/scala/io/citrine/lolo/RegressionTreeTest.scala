package io.citrine.lolo

import io.citrine.lolo.trees.RegressionTreeLearner
import org.junit.Test

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class RegressionTreeTest {


  @Test
  def testSimpleTree(): Unit = {
    val csv = TestUtils.readCsvAsDoubles("double_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last))
    val DTLearner = new RegressionTreeLearner()
    val DT = DTLearner.train(trainingData.asInstanceOf[Seq[(Array[Any], Double)]])
    trainingData.foreach { case (x, y) =>
        assert(y == DT.predict(x.asInstanceOf[Array[Any]]))
    }
  }

  @Test
  def longerTest(): Unit = {
    val csv = TestUtils.readCsvAsDoubles("large_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last))
    val DTLearner = new RegressionTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DT = DTLearner.train(trainingData.asInstanceOf[Seq[(Array[Any], Double)]])
    (0 until N).map(i => DTLearner.train(trainingData.asInstanceOf[Seq[(Array[Any], Double)]]))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration/N} s")

    /* We should be able to memorize the inputs */
    trainingData.foreach { case (x, y) =>
        assert(y == DT.predict(x.asInstanceOf[Array[Any]]))
    }

    /* The first feature should be the most important
    val importances = DT.getFeatureImportance()
    println(importances.toList)
    assert(importances(0) == importances.max)
    */
  }

}

object RegressionTreeTest {
  def main(argv: Array[String]): Unit = {
    // new SimpleRegressionTreeTest().testSimpleTree()
    new RegressionTreeTest().longerTest()
  }
}
