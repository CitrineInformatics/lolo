package io.citrine.lolo.trees.multitask

import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.classification.ClassificationTree
import io.citrine.lolo.trees.regression.RegressionTree
import org.junit.Test
import org.scalatest.Assertions._

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class MultiTaskTreeTest {

  @Test
  def testTwoLabels(): Unit = {
    val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)
    val inputs = raw.map(_._1)
    val realLabel = raw.map(_._2)
    val catLabel = raw.map(_._2 > realLabel.max / 2.0)
    val learner = new MultiTaskTreeLearner()
    val models = learner.train(inputs, Seq(realLabel, catLabel)).map(_.getModel())
    assert(models.size == 2)
    assert(models.head.isInstanceOf[RegressionTree])
    assert(models.last.isInstanceOf[ClassificationTree])
    val realResults = models.head.transform(inputs).getExpected().asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]

    assert(realLabel.zip(realResults).forall(p => p._1 == p._2))
    assert(catLabel.zip(catResults).forall(p => p._1 == p._2))
  }
}


