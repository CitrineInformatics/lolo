package io.citrine.lolo

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{BoltzmannSplitter, RegressionSplitter}
import io.citrine.theta.Stopwatch
import org.junit.Test

/**
  * Created by maxhutch on 7/10/17.
  */
class AccuracyTest {

  val noiseLevel = 0.000
  val nFeat = 10
  val nRow = 128
  val nScal = 2
  val trainingData = TestUtils.binTrainingData(TestUtils.generateTrainingData(nRow, nFeat, noise = noiseLevel, seed = 3L), inputBins = Seq((4, 32)))
  val minInstances = 1

  private def computeMetrics(learner: Learner): Double = {
    val pva = learner.train(trainingData).getLoss().get
    pva
  }

  @Test
  def testRandomForest(): Unit = {
    val baseLearner = new RegressionTreeLearner(numFeatures = nFeat / 3, minLeafInstances = minInstances)
    val learner = new Bagger(baseLearner, numBags = nRow * nScal)
    val error = computeMetrics(learner)
    assert(error > noiseLevel, s"Can't do better than noise")
    println(s"Error in RF is ${error}")
  }

  @Test
  def testLowTLimit(): Unit = {
    println("Hello?")
    val errorStandardTree = {
      val baseLearner = new RegressionTreeLearner(numFeatures = nFeat)
      val learner = new Bagger(baseLearner, numBags = nRow * 16)
      println(s"Normal train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner)
    }
    val errorAnnealingTree = {
      val baseLearner = new RegressionTreeLearner(splitter = BoltzmannSplitter(temperature = 1.0e-9))
      val learner = new Bagger(baseLearner, numBags = nRow * 16)
      println(s"Annealing train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner)
    }

    val relativeDifference = 2.0 * Math.abs(errorAnnealingTree - errorStandardTree)/(errorAnnealingTree + errorStandardTree)
    println(relativeDifference)
    assert(relativeDifference < 0.01)
  }

  @Test
  def testLessRandomForest(): Unit = {
    val baseLearner = new RegressionTreeLearner(numFeatures = nFeat, minLeafInstances = minInstances)
    val learner = new Bagger(baseLearner, numBags = nRow * nScal)
    val error = computeMetrics(learner)
    assert(error > noiseLevel, s"Can't do better than noise")
    println(s"Error in LRF is ${error}")
  }

  def testAnnealingForest(): Unit = {
    var temp = 0.001
    var base: Double = 0.0
    val improvements: Seq[(Double, Double)] = (0 until 32).map { i =>
      val baseLearner = new RegressionTreeLearner(splitter = BoltzmannSplitter(temperature = temp), minLeafInstances = minInstances)
      val learner = new Bagger(baseLearner, numBags = nRow * nScal)
      val error = computeMetrics(learner)
      if (base == 0.0) base = error
      assert(error > noiseLevel, s"Can't do better than noise")
      println(f"Error in annealing tree at T=${temp}%6.2e is ${error}%5.2f (${error / base}%5.3f)")
      val res = (temp, base / error)
      temp = temp * Math.pow(2.0, 1.0/4.0)
      res
    }
    val best = improvements.maxBy(_._2)
    println(f"The best improvement was ${best._2}%5.2f x at T=${best._1}%6.2e")
  }


}

object AccuracyTest {

  val trainingDataFull: Seq[(Vector[Any], Double)] = TestUtils.binTrainingData(TestUtils.generateTrainingData(2048, 48), inputBins = Seq((2, 32)))
    .asInstanceOf[Seq[(Vector[Any], Double)]]

  def computeMetrics(nRow: Int, nFeat: Int, nFeatSub: Int, nScal: Int, minInstances: Int, temperature: Double): (Double, Double) = {
    val trainingData = trainingDataFull.take(nRow).map{case (f, l) => (f.take(nFeat), l)}
    val splitter = if (temperature > 0) {
      BoltzmannSplitter(temperature = temperature)
    } else {
      RegressionSplitter()
    }
    val baseLearner = new RegressionTreeLearner(
      numFeatures = nFeatSub,
      splitter = splitter,
      minLeafInstances = minInstances
    )
    val learner = new Bagger(baseLearner, numBags = nRow * nScal, biasLearner = None)
    val model = learner.train(trainingData).getModel()
    val (features, labels) = trainingData.unzip
    val predictions = model.transform(features)
    val expected = predictions.getExpected().asInstanceOf[Seq[Double]]
    val sigma = predictions.getUncertainty().get.asInstanceOf[Seq[Double]]
    val pva: Seq[(Double, Double, Double)] = labels.indices.map{i =>
      (labels(i), expected(i), sigma(i))
    }
    val rmse = Math.sqrt(pva.map{case (x, y, s) => Math.pow(x-y, 2)}.sum / pva.size)
    val stdres = Math.sqrt(pva.map{case (x, y, s) => Math.pow((x-y) / s, 2)}.sum / pva.size)
    (rmse, stdres)
  }


  def main(args: Array[String]): Unit = {
    var temp = 0.002
    val nRow = 256
    val nFeat = 48
    val nScal = 1
    val minInstances = 1
    val (baseRMSE: Double, baseStdRes: Double) = computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temperature = 0.0)
    println(baseRMSE, baseStdRes)
    val improvements: Seq[(Double, Double, Double)] = (0 until 32).map { i =>
      val (error, stdres) = computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temp)
      println(f"Error in annealing tree at T=${temp}%6.2e is ${error}%5.2f (${error / baseRMSE}%5.3f, ${stdres}%5.3f)")
      val res = (temp, baseRMSE / error, baseStdRes / stdres)
      temp = temp * Math.pow(2.0, 1.0/4.0)
      res
    }
    val best = improvements.maxBy(_._2)
    println(f"The best improvement was ${best._2}%5.2f x at T=${best._1}%6.2e")
    val bestt = improvements.maxBy(_._3)
    println(f"The best improvement was ${bestt._3}%5.2f x at T=${bestt._1}%6.2e")
  }
}
