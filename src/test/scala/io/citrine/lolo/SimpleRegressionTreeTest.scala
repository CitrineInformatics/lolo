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
    val csv = TestUtils.readCsvAsDoubles("double_example.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last))
    val DTLearner = new SimpleRegressionTreeLearner()
    val DT = DTLearner.train(trainingData)
    trainingData.foreach { case (x, y) =>
        assert(y == DT.predict(x))
    }
  }

}

object SimpleRegressionTreeTest {
  def main(argv: Array[String]): Unit = {
    new SimpleRegressionTreeTest().testSimpleTree()
  }
}
