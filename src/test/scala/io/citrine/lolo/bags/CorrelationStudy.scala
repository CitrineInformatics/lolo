package io.citrine.lolo.bags

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.TestUtils
import io.citrine.lolo.bags.CorrelationMethods.CorrelationMethod
import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.stats.utils
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

case class TrialResult(
                      predictionResult: MultiTaskBaggedResult,
                      trueLabels: Seq[Seq[Double]],
                      observedLabels: Seq[Seq[Double]]
                      )

object CorrelationStudy {

  def main(args: Array[String]): Unit = {
    val seed = 52109317L
    val mainRng = new Random(seed)

    val numTrials = 24
    val numTrainRows = 128
    val numBags = numTrainRows
    val numTestRows = 128
    val numCols = 12
    val samplingNoise = 0.0
    val linearRho = 0.98
    val quadraticCorrelationFuzz = 0.0

    val allResults = Seq.fill(numTrials) {
      val rng = new Random(mainRng.nextLong())
      val dataGenSeed = rng.nextLong()
      val dataNoiseSeed = new Random(rng.nextLong())
      val trainRng = new Random(rng.nextLong())
      val bagSeed = rng.nextLong()
      val linearCorrelationRng = new Random(rng.nextLong())
      val quadraticCorrelationRng = new Random(rng.nextLong())

      val fullData: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(
        numTrainRows + numTestRows,
        numCols,
        noise = 0.0, // Add noise later, after computing covariate labels
        function = Friedman.friedmanSilverman,
        seed = dataGenSeed
      )

      val inputs: Seq[Vector[Double]] = fullData.map(_._1)
      val realLabel: Seq[Double] = fullData.map(_._2)
      val linearLabel: Seq[Double] = makeLinearCorrelatedData(realLabel, linearRho, linearCorrelationRng)
      val quadraticLabel: Seq[Double] = makeQuadraticCorrelatedData(realLabel, quadraticCorrelationFuzz, quadraticCorrelationRng)

      val realLabelNoise = realLabel.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())
      val linearLabelNoise = linearLabel.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())
      val quadraticLabelNoise = quadraticLabel.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())

      val learner = MultiTaskTreeLearner(rng = trainRng)
      val baggedLearner = MultiTaskBagger(
        learner,
        numBags =  numBags,
        randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(bagSeed)))
      )

      val RF = baggedLearner.train(
        inputs.take(numTrainRows),
        Seq(realLabelNoise.take(numTrainRows), linearLabelNoise.take(numTrainRows), quadraticLabelNoise.take(numTrainRows))
      ).getModel()

      val predictionResult = RF.transform(inputs.drop(numTrainRows)).asInstanceOf[MultiTaskBaggedResult]

      val trueLabels = Seq(
        realLabel.drop(numTrainRows),
        linearLabel.drop(numTrainRows),
        quadraticLabel.drop(numTrainRows)
      )
      val observedLabels = Seq(
        realLabelNoise.drop(numTrainRows),
        linearLabelNoise.drop(numTrainRows),
        quadraticLabelNoise.drop(numTrainRows)
      )
      TrialResult(predictionResult, trueLabels, observedLabels)
    }

    for (index <- Seq(1, 2)) {
      for (observational <- Seq(true, false)) {
        for (method <- Seq(CorrelationMethods.Trivial, CorrelationMethods.Bootstrap, CorrelationMethods.Jackknife)) {
          val nlpd = aggregateTrialResults(allResults, 0, index, method, observational)
          val nlpdMean = utils.mean(nlpd)
          val nlpdStdErr = math.sqrt(utils.variance(nlpd, dof = 1) / nlpd.length)
          val dataType = if (index == 1) "linear correlated data" else "quadratic correlated data"
          val errorType = if (observational) "observational error" else "true error"
          println(s"$dataType, $method, $errorType, nlpd = $nlpdMean +/- $nlpdStdErr")
        }
      }
    }
  }

  def aggregateTrialResults(trialResults: Seq[TrialResult], i: Int, j: Int, method: CorrelationMethod, observational: Boolean): Seq[Double] = {
    trialResults.map { trialResult =>
      val predictions = trialResult.predictionResult.getExpected()
      val predictionsI = predictions.map(x => x(i)).asInstanceOf[Seq[Double]]
      val predictionsJ = predictions.map(x => x(j)).asInstanceOf[Seq[Double]]
      val (actualI, actualJ) = if (observational) {
        (trialResult.observedLabels(i), trialResult.observedLabels(j))
      } else {
        (trialResult.trueLabels(i), trialResult.trueLabels(j))
      }
      val errorI = predictionsI.zip(actualI).map(x => x._1 - x._2)
      val errorJ = predictionsJ.zip(actualJ).map(x => x._1 - x._2)
      val rho = trialResult.predictionResult.getUncertaintyCorrelationBuffet(i, j, method).get
      val allSigma = trialResult.predictionResult.getUncertainty(observational)
      val sigmaI = allSigma.get.map(x => x(i).getOrElse(0.0)).asInstanceOf[Seq[Double]]
      val sigmaJ = allSigma.get.map(x => x(j).getOrElse(0.0)).asInstanceOf[Seq[Double]]
      val nlpd = NLPD(errorI, errorJ, sigmaI, sigmaJ, rho)
      utils.median(nlpd)
    }
  }

  def NLPD(dx: Seq[Double], dy: Seq[Double], sigmaX: Seq[Double], sigmaY: Seq[Double], rho: Seq[Double]): Seq[Double] = {
    Seq(dx, dy, sigmaX, sigmaY, rho).transpose.map { foo =>
      val x = foo.head
      val y = foo(1)
      val sx = foo(2)
      val sy = foo(3)
      val r = foo(4)
      val r2 = math.pow(r, 2.0)
      val term1 = math.log(2 * math.Pi * sx * sy * math.sqrt(1 - r2))
      val normX = x / sx
      val normY = y / sy
      val term2 = 1/(2 * (1 - r2)) * (math.pow(normX, 2.0) + math.pow(normY, 2.0) - 2 * r * normX * normY)
      term1 + term2
    }
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

  def makeQuadraticCorrelatedData(X: Seq[Double], fuzz: Double = 0.0, rng: Random = new Random()): Seq[Double] = {
    require(fuzz >= 0.0)
    val mu = X.sum / X.size
    X.map(x => math.pow(x - mu, 2.0) + fuzz * rng.nextGaussian())
  }

}