package io.citrine.lolo

import io.citrine.lolo.bags.{Bagger, RegressionBagger}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.{BoltzmannSplitter, RegressionSplitter}
import io.citrine.random.Random
import org.junit.Test

/**
  * Created by maxhutch on 7/10/17.
  */
class AccuracyTest extends SeedRandomMixIn {

  val noiseLevel: Double = 0.000
  val nFeat: Int = 10
  val nRow: Int = 128
  val nScal: Int = 2
  val minInstances: Int = 1

  val trainingData: Seq[TrainingRow[Double]] =
    DataGenerator.generate(nRow, nFeat, noise = noiseLevel, rng = rng).withBinnedInputs(bins = Seq((4, 32))).data

  // Get the out-of-bag RMSE
  private def computeMetrics(learner: Learner[Double], rng: Random): Double = {
    learner.train(trainingData, rng = rng).loss.get
  }

  /**
    * Quick sanity check of the test setup
    */
  @Test
  def testRandomForest(): Unit = {
    val baseLearner = RegressionTreeLearner(numFeatures = nFeat / 3, minLeafInstances = minInstances)
    val learner = RegressionBagger(baseLearner, numBags = nRow * nScal)
    val error = computeMetrics(learner, rng)
    assert(error > noiseLevel, s"Can't do better than noise")
    assert(error < 4.0, "Error increased, probably due to a change in configuration")
  }

  /**
    * Check that a low-temperature Boltzmann Tree recovers the performance of a normal RF (with randomized pivots)
    */
  @Test
  def testLowTLimit(): Unit = {
    val errorStandardTree = {
      val baseLearner = RegressionTreeLearner(
        numFeatures = nFeat,
        splitter = RegressionSplitter(randomizePivotLocation = true)
      )
      val learner = RegressionBagger(baseLearner, numBags = nRow * 16)
      // println(s"Normal train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner, rng)
    }
    val errorAnnealingTree = {
      val baseLearner = RegressionTreeLearner(
        numFeatures = nFeat,
        splitter = BoltzmannSplitter(temperature = Float.MinPositiveValue)
      )
      val learner = RegressionBagger(baseLearner, numBags = nRow * 16)
      // println(s"Annealing train time: ${Stopwatch.time(computeMetrics(learner))}")
      computeMetrics(learner, rng)
    }

    val relativeDifference =
      2.0 * Math.abs(errorAnnealingTree - errorStandardTree) / (errorAnnealingTree + errorStandardTree)
    // println(relativeDifference)
    assert(relativeDifference < 0.01)
  }
}

/**
  * Driver code to study the performance vs temperature
  *
  * This isn't cast as a test, but can be used to try to understand the behavior of Boltzmann trees on some simple problems.
  */
object AccuracyTest extends SeedRandomMixIn {

  val trainingDataFull: Seq[TrainingRow[Double]] =
    DataGenerator.generate(2048, 48, rng = rng).withBinnedInputs(bins = Seq((2, 32))).data

  /**
    * Compute the RMSE and standard residual for a Boltzmann tree with the given temperature
    * @param nRow training set size
    * @param nFeat number of features
    * @param nFeatSub number of features considered per split
    * @param nScal number of trees, as a multiple of the training set size
    * @param minInstances minimum tree node size
    * @param temperature of the boltzmann tree
    * @return (RMSE, standard residual)
    */
  def computeMetrics(
      nRow: Int,
      nFeat: Int,
      nFeatSub: Int,
      nScal: Int,
      minInstances: Int,
      temperature: Double
  ): (Double, Double) = {
    val trainingData = trainingDataFull.take(nRow).map(_.mapInputs(inputs => inputs.take(nFeat)))
    val splitter = if (temperature > 0) {
      BoltzmannSplitter(temperature = temperature)
    } else {
      RegressionSplitter(randomizePivotLocation = true)
    }
    val baseLearner = RegressionTreeLearner(
      numFeatures = nFeatSub,
      splitter = splitter,
      minLeafInstances = minInstances
    )
    val learner = RegressionBagger(baseLearner, numBags = nRow * nScal, biasLearner = None)
    val model = learner.train(trainingData, rng = rng).model
    val (features, labels) = trainingData.map(row => (row.inputs, row.label)).unzip
    val predictions = model.transform(features)
    val expected = predictions.expected
    val sigma = predictions.uncertainty().get.asInstanceOf[Seq[Double]]
    val pva: Seq[(Double, Double, Double)] = labels.indices.map { i =>
      (labels(i), expected(i), sigma(i))
    }
    val rmse = Math.sqrt(pva.map { case (x, y, _) => Math.pow(x - y, 2) }.sum / pva.size)
    val stdres = Math.sqrt(pva.map { case (x, y, s) => Math.pow((x - y) / s, 2) }.sum / pva.size)
    (rmse, stdres)
  }

  def main(args: Array[String]): Unit = {
    var temp = 0.00001
    val nRow = 256
    val nFeat = 48
    val nScal = 1
    val minInstances = 1
    val (baseRMSE: Double, baseStdRes: Double) =
      computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temperature = 0.0)
    println(baseRMSE, baseStdRes)
    val improvements: Seq[(Double, Double, Double)] = (0 until 64).map { i =>
      val (error, stdres) = computeMetrics(nRow, nFeat, nFeat, nScal, minInstances, temp)
      println(f"Error in annealing tree at T=${temp}%6.2e is ${error}%5.2f (${error / baseRMSE}%5.3f, ${stdres}%5.3f)")
      val res = (temp, baseRMSE / error, baseStdRes / stdres)
      temp = temp * Math.pow(2.0, 1.0 / 4.0)
      res
    }
    val best = improvements.maxBy(_._2)
    println(f"The best improvement was ${best._2}%5.2f x at T=${best._1}%6.2e")
    val bestt = improvements.maxBy(_._3)
    println(f"The best improvement was ${bestt._3}%5.2f x at T=${bestt._1}%6.2e")
  }
}
