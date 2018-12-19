package io.citrine.lolo.learners

import io.citrine.lolo.TestUtils
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

/**
  * Created by maxhutch on 1/9/17.
  */
@Test
class RandomForestTest {

  /**
    * Test that the regression forest does the same thing as the regression bagger
    */
  @Test
  def testRegressionForest(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )

    val RFMeta = RandomForest()
      .train(trainingData)
    val RF = RFMeta.getModel()

    assert(RFMeta.getLoss().get < 1.0, "Loss of bagger is larger than expected")

    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
    assert(sigma.forall(_ >= 0.0))

    assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")

    /* The first feature should be the most important */
    val importances = RFMeta.getFeatureImportance().get
    assert(importances(1) == importances.max)
  }

  /**
    * Test that the regression forest does the same thing as the classification bagger
    */
  @Test
  def testClassificationForest(): Unit = {
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )
    val RFMeta = new RandomForest(numTrees = trainingData.size * 2)
      .train(trainingData)
    val RF = RFMeta.getModel()

    /* Inspect the results */
    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected()
    assert(trainingData.map(_._2).zip(means).forall { case (a, p) => a == p })

    val uncertainty = results.getUncertainty()
    assert(uncertainty.isDefined)
    assert(trainingData.map(_._2).zip(uncertainty.get).forall { case (a, probs) =>
      val classProbabilities = probs.asInstanceOf[Map[Any, Double]]
      val maxProb = classProbabilities(a)
      maxProb > 0.5 && maxProb < 1.0 && Math.abs(classProbabilities.values.sum - 1.0) < 1.0e-6
    })
  }

  /**
    * Test that the regression forest does the same thing as the regression bagger
    */
  @Test
  def testTreeConvergence(): Unit = {
    val nRow = 256
    val (trainingData, testData) = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(nRow * 2, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    ).splitAt(nRow)

    println(Runtime.getRuntime.totalMemory() / Math.pow(2, 20))

    val (ucurve, ecurve) = Seq.tabulate(13){idx => (nRow * Math.pow(2, 7 - idx)).toInt}.map{nTree =>
      val RFMeta = RandomForest(numTrees = nTree).train(trainingData)
      val RF = RFMeta.getModel()
      val results = RF.transform(testData.map(_._1))
      val sigma: Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Double]]
      // assert(sigma.forall(_ >= 0.0))
      assert(results.getGradient().isEmpty, "Returned a gradient when there shouldn't be one")

      /* The first feature should be the most important */
      val importances = RFMeta.getFeatureImportance().get
      assert(importances(1) == importances.max)

      val totalUncertainty = sigma.sum
      val totalError = results.getExpected().zip(testData.map(_._2)).map{case (x: Double, y: Double) => Math.abs(x - y)}.sum
      println(nTree, totalUncertainty)
      ((nTree, totalUncertainty), (nTree, totalError))
    }.unzip
    println(ecurve)
    println(ucurve)
    val adjusted = ucurve.tail.map{case (idx, u) => (Math.log(idx), Math.log(u - ucurve.head._2))}
    println(adjusted.map{x => f"(${x._1}%4.2f, ${x._2}%4.2f)"}.foldLeft(""){case (x: String, y: String) => s"${x}, ${y}"})
    val slopes = adjusted.indices.tail.map{idx =>
      (adjusted(idx)._2 - adjusted(idx - 1)._2) / (adjusted(idx)._1 - adjusted(idx - 1)._1)
    }
    println(slopes.toVector)
  }

}

object RandomForestTest {
  def main(argv: Array[String]): Unit = {
    new RandomForestTest()
      .testClassificationForest()
  }
}
