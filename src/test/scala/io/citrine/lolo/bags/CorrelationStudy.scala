package io.citrine.lolo.bags

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.TestUtils
import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.validation.{Merit, NegativeLogProbabilityDensity2d}
import org.apache.commons.math3.random.MersenneTwister
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.BitmapEncoder.BitmapFormat

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

    val numTrials = 10
    val numTrainRows = 128
    val numBags = numTrainRows
    val numTestRows = 128
    val numCols = 12
    val samplingNoise = 0.0
    val quadraticCorrelationFuzz = 0.0

    val chart = Merit.plotMeritScan(
      parameterName = "Strength of training data correlation",
      parameterValues = Seq(0.0, 0.25, 0.5, 0.75, 0.9, 0.99),
      merits = Map(
        "Trivial" -> NegativeLogProbabilityDensity2d(0, 1, CorrelationMethods.Trivial, true),
        "Training Data" -> NegativeLogProbabilityDensity2d(0, 1, CorrelationMethods.FromTraining, true),
        "Bootstrap" -> NegativeLogProbabilityDensity2d(0, 1, CorrelationMethods.Bootstrap, true),
        "Jackknife" -> NegativeLogProbabilityDensity2d(0, 1, CorrelationMethods.Jackknife, true)
      ),
      rng = new Random(7216357L)
    ) { rhoTrain =>
      Iterator.tabulate(numTrials) { _ =>
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
        val linearLabel: Seq[Double] = makeLinearCorrelatedData(realLabel, rhoTrain, linearCorrelationRng)
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
        ).transpose
        val observedLabels = Seq(
          realLabelNoise.drop(numTrainRows),
          linearLabelNoise.drop(numTrainRows),
          quadraticLabelNoise.drop(numTrainRows)
        ).transpose
        (predictionResult, observedLabels)
      }
    }

    val fname = "./scan-rho-observed-error"
    BitmapEncoder.saveBitmap(chart, fname, BitmapFormat.PNG)
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