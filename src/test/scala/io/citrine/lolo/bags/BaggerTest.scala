package io.citrine.lolo.bags

import io.citrine.lolo.TestUtils
import io.citrine.lolo.trees.RegressionTreeLearner
import org.junit.Test

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class BaggerTest {

  @Test
  def testRegressionBagger(): Unit = {
    val csv = TestUtils.readCsv("large_example_with_cat.csv")
    val trainingData = csv.map(vec => (vec.init, vec.last.asInstanceOf[Double]))
    val DTLearner = new RegressionTreeLearner()
    val baggedLearner = new Bagger(DTLearner)
    val N = 0
    val start = System.nanoTime()
    val RF = baggedLearner.train(trainingData)
    (0 until N).map(i => baggedLearner.train(trainingData))
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration/(N + 1)} s")

    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    val uncertainties = results.getUncertainty()

    /* The first feature should be the most important */
    val importances = RF.getFeatureImportance()
    assert(importances(0) == importances.max)
  }
}

object BaggerTest {
  def main(argv: Array[String]): Unit = {
    new BaggerTest().testRegressionBagger()
  }
}
