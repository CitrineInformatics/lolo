package io.citrine.lolo.trees.multitask

import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.metrics.ClassificationMetrics
import io.citrine.lolo.trees.classification.{ClassificationTree, ClassificationTreeLearner}
import io.citrine.lolo.trees.regression.RegressionTree
import org.junit.Test
import org.scalatest.Assertions._

import scala.util.Random

/**
  * Created by maxhutch on 11/29/16.
  */
@Test
class MultiTaskTreeTest {

  /* Setup some data */
  val raw: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman)
  val inputs: Seq[Vector[Double]] = raw.map(_._1)
  val realLabel: Seq[Double] = raw.map(_._2)
  val catLabel: Seq[Boolean] = raw.map(_._2 > realLabel.max / 2.0)

  /**
    * Test that the tree memorizes the data when it is complete
    */
  @Test
  def testTwoLabels(): Unit = {
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

  /**
    * Test that the tree memorizes the data when it is sparse
    */
  @Test
  def testSparseExact(): Unit = {
    val sparseCat = catLabel.map(x =>
      if (Random.nextBoolean()) {
        null
      } else {
        x
      }
    )

    val learner = new MultiTaskTreeLearner()
    val models = learner.train(inputs, Seq(realLabel, sparseCat)).map(_.getModel())
    val realResults = models.head.transform(inputs).getExpected().asInstanceOf[Seq[Double]]
    val catResults = models.last.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]

    assert(realLabel.zip(realResults).forall(p => p._1 == p._2))
    assert(sparseCat.zip(catResults).forall(p => p._1 == null || p._1 == p._2))
  }

  /**
    * Test that the multitask tree on sparse boolean labels is better than single task
    */
  @Test
  def testSparseQuantitative(): Unit = {
    val sparseCat = catLabel.map(x =>
      if (Random.nextDouble() > 0.125) {
        null
      } else {
        x
      }
    )

    val learner = new MultiTaskTreeLearner()
    val models = learner.train(inputs, Seq(realLabel, sparseCat)).map(_.getModel())
    val catResults = models.last.transform(inputs).getExpected().asInstanceOf[Seq[Boolean]]

    val reference = new ClassificationTreeLearner()
      .train(inputs.zip(sparseCat).filterNot(_._2 == null))
      .getModel()
      .transform(inputs)
      .getExpected()

    val singleF1 = ClassificationMetrics.f1scores(reference, catLabel)
    val multiF1 = ClassificationMetrics.f1scores(catResults, catLabel)
    assert(multiF1 < 1.0)
    assert(multiF1 > singleF1)
  }

}


