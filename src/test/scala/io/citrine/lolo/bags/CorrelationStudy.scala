package io.citrine.lolo.bags

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

object CorrelationStudy {

  def main(args: Array[String]): Unit = {
    val seed = 52109317L
    val rng = new Random(seed)
    val trainSeed = rng.nextLong()
    val testSeed = rng.nextLong()

    val numBags = 64
    val numTrainRows = 128
    val numTestRows = 32
    val numCols = 12
    val samplingNoise = 0.0
    val linearRho = 1.0
    val quadraticCorrelationNoise = 0.0

    val rawTrainingData: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(
      numTrainRows,
      numCols,
      noise = samplingNoise,
      function = Friedman.friedmanSilverman,
      seed = trainSeed
    )
    val inputs: Seq[Vector[Double]] = rawTrainingData.map(_._1)
    val realLabel: Seq[Double] = rawTrainingData.map(_._2)
    val catLabel: Seq[Boolean] = rawTrainingData.map(_._2 > realLabel.max / 2.0)
    val linearCorrelatedLabel: Seq[Double] = makeLinearCorrelatedData(realLabel, linearRho, rng)
    val quadraticCorrelatedLabel: Seq[Double] = makeQuadraticCorrelatedData(realLabel, quadraticCorrelationNoise, rng)

    val learner = MultiTaskTreeLearner(rng = rng)
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags =  numBags,
      randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(rng.nextLong())))
    )
    val RF = baggedLearner.train(inputs, Seq(realLabel, catLabel, linearCorrelatedLabel, quadraticCorrelatedLabel)).getModel()

    val testInputs = TestUtils
      .generateTrainingData(numTestRows, numCols, function = Friedman.friedmanSilverman, seed = testSeed)
      .map(_._1)

    val predictionResult = RF.transform(testInputs).asInstanceOf[MultiTaskBaggedResult]
    val shouldBe1 = predictionResult.getUncertaintyCorrelationBuffet(0, 0, CorrelationMethods.Trivial)
    val shouldBeNone = predictionResult.getUncertaintyCorrelationBuffet(0, 1, CorrelationMethods.Trivial)
    val shouldBe0 = predictionResult.getUncertaintyCorrelationBuffet(0, 2, CorrelationMethods.Trivial)

    val shouldBeRho = predictionResult.getUncertaintyCorrelationBuffet(0, 2, CorrelationMethods.FromTraining)
    val shouldBeBasicallyZero = predictionResult.getUncertaintyCorrelationBuffet(0, 3, CorrelationMethods.FromTraining)

    val bootstrap1 = predictionResult.getUncertaintyCorrelationBuffet(0, 2, CorrelationMethods.Bootstrap)
    val bootstrap2 = predictionResult.getUncertaintyCorrelationBuffet(0, 3, CorrelationMethods.Bootstrap)

    val jackknife1 = predictionResult.getUncertaintyCorrelationBuffet(0, 2, CorrelationMethods.Jackknife)
    val jackknife2 = predictionResult.getUncertaintyCorrelationBuffet(0, 3, CorrelationMethods.Jackknife)
  }

  def makeLinearCorrelatedData(X: Seq[Double], rho: Double, rng: Random = new Random()): Seq[Double] = {
    require(rho >= -1.0 && rho <= 1.0)
    val Y = Seq.fill(X.length)(rng.nextGaussian())
    val linearLearner = LinearRegressionLearner()
    val linearModel = linearLearner.train(X.zip(Y).map { case (x, y) => (Vector(x), y) } ).getModel()
    val yPred = linearModel.transform(X.map(Vector(_))).getExpected()
    val residuals = Y.zip(yPred).map { case (actual, predicted) => actual - predicted }
    val muX = X.sum / X.size
    val stdX = math.sqrt(X.map(a => math.pow(a - muX, 2.0)).sum / X.size)
    val muResiduals = residuals.sum / residuals.size
    val stdResiduals = math.sqrt(residuals.map(a => math.pow(a - muResiduals, 2.0)).sum / residuals.size)
    X.zip(residuals).map { case (x, residual) => rho * stdResiduals * x + math.sqrt(1 - rho * rho) * stdX * residual}
  }

  def makeQuadraticCorrelatedData(X: Seq[Double], sigma: Double = 0.0, rng: Random = new Random()): Seq[Double] = {
    require(sigma >= 0.0)
    val mu = X.sum / X.size
    X.map(x => math.pow(x - mu, 2.0) + sigma * rng.nextGaussian())
  }

}