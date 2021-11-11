package io.citrine.lolo.bags

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.{PredictionResult, TestUtils}
import io.citrine.lolo.stats.StatsUtils.makeLinearCorrelatedData
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.validation.{Merit, NegativeLogProbabilityDensity2d, StandardConfidence2d}
import org.apache.commons.math3.random.MersenneTwister
import org.knowm.xchart.{BitmapEncoder, XYChart}
import org.knowm.xchart.BitmapEncoder.BitmapFormat

import scala.util.Random

sealed trait TestProblems
case object Linear extends TestProblems
case object Quadratic extends TestProblems

sealed trait Metric
case object NLPD extends Metric
case object StdConfidence extends Metric

sealed trait VariedParameter {
  def name: String
}
case object TrainRho extends VariedParameter {
  def name: String = "linear training correlation"
}
case object TrainQuadraticFuzz extends VariedParameter {
  def name: String = "shift to decorrelate quadratic data"
}
case object Noise extends VariedParameter {
  def name: String = "observational noise level"
}
case object Bags extends VariedParameter {
  def name: String = "number of bags"
}
case object NumTraining extends VariedParameter {
  def name: String = "number of training rows"
}

object CorrelationStudy {

  def main(args: Array[String]): Unit = {
    val mainRng = new Random(52109317L)

    val numTrials = 10
    val numTrainRows = 128
    val numTestRows = 128
    val numCols = 12

    makeAndSaveChart(
      fname = "./test",
      metric = NLPD,
      variedParameter = TrainRho,
      parameterValues = Seq(0.0, 0.25, 0.50, 0.75, 0.9, 0.99),
      testProblem = Linear,
      function = Friedman.friedmanSilverman,
      numTrials = numTrials,
      numTrain = numTrainRows,
      numTest = numTestRows,
      numCols = numCols,
      observational = true,
      samplingNoise = 0.0,
      rhoTrain = 0.0,
      quadraticCorrelationFuzz = 0.0,
      rng = mainRng
    )
  }

  def makeAndSaveChart(
                        fname: String,
                        metric: Metric,
                        variedParameter: VariedParameter,
                        parameterValues: Seq[Double],
                        testProblem: TestProblems,
                        function: Seq[Double] => Double,
                        numTrials: Int,
                        numTrain: Int,
                        numTest: Int,
                        numCols: Int,
                        observational: Boolean,
                        samplingNoise: Double,
                        rhoTrain: Double,
                        quadraticCorrelationFuzz: Double,
                        rng: Random
                      ): Unit = {
    // TODO: also save the raw data in a csv
    val chart = makeChart(
      variedParameter = variedParameter,
      metric = metric,
      parameterValues = parameterValues,
      testProblem = testProblem,
      function = function,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      numCols = numCols,
      observational = observational,
      samplingNoise = samplingNoise,
      rhoTrain = rhoTrain,
      quadraticCorrelationFuzz = quadraticCorrelationFuzz,
      rng = rng
    )
    BitmapEncoder.saveBitmap(chart, fname + ".png", BitmapFormat.PNG)
  }

  def makeChart(
                 variedParameter: VariedParameter,
                 metric: Metric,
                 parameterValues: Seq[Double],
                 testProblem: TestProblems,
                 function: Seq[Double] => Double,
                 numTrials: Int,
                 numTrain: Int,
                 numTest: Int,
                 numCols: Int,
                 observational: Boolean,
                 samplingNoise: Double,
                 rhoTrain: Double,
                 quadraticCorrelationFuzz: Double,
                 rng: Random
               ): XYChart = {
    val index: Int = testProblem match {
      case Linear => 1
      case Quadratic => 2
    }
    val merits = metric match {
      case NLPD =>
        Map(
          "Trivial" -> NegativeLogProbabilityDensity2d(0, index, CorrelationMethods.Trivial, observational),
          "Training Data" -> NegativeLogProbabilityDensity2d(0, index, CorrelationMethods.FromTraining, observational),
          "Bootstrap" -> NegativeLogProbabilityDensity2d(0, index, CorrelationMethods.Bootstrap, observational),
          "Jackknife" -> NegativeLogProbabilityDensity2d(0, index, CorrelationMethods.Jackknife, observational)
        )
      case StdConfidence =>
        Map(
          "Trivial" -> StandardConfidence2d(0, index, CorrelationMethods.Trivial, observational),
          "Training Data" -> StandardConfidence2d(0, index, CorrelationMethods.FromTraining, observational),
          "Bootstrap" -> StandardConfidence2d(0, index, CorrelationMethods.Bootstrap, observational),
          "Jackknife" -> StandardConfidence2d(0, index, CorrelationMethods.Jackknife, observational)
        )
    }

    val pvaBuilder: Double => Iterator[(PredictionResult[Seq[Any]], Seq[Seq[Any]])] = variedParameter match {
      case TrainRho =>
        rho => Iterator.tabulate(numTrials) { _ =>
          val thisRng = new Random(rng.nextLong())
          runTrial(
            function = function,
            numTrain = numTrain,
            numTest = numTest,
            numCols = numCols,
            numBags = numTrain,
            rhoTrain = rho,
            quadraticCorrelationFuzz = quadraticCorrelationFuzz,
            samplingNoise = samplingNoise,
            observational = observational,
            rng = thisRng
          )
        }
      case Noise =>
        noiseLevel => Iterator.tabulate(numTrials) { _ =>
          val thisRng = new Random(rng.nextLong())
          runTrial(
            function = function,
            numTrain = numTrain,
            numTest = numTest,
            numCols = numCols,
            numBags = numTrain,
            rhoTrain = rhoTrain,
            quadraticCorrelationFuzz = quadraticCorrelationFuzz,
            samplingNoise = noiseLevel,
            observational = observational,
            rng = thisRng
          )
        }
      case TrainQuadraticFuzz =>
        fuzz => Iterator.tabulate(numTrials) { _ =>
          val thisRng = new Random(rng.nextLong())
          runTrial(
            function = function,
            numTrain = numTrain,
            numTest = numTest,
            numCols = numCols,
            numBags = numTrain,
            rhoTrain = rhoTrain,
            quadraticCorrelationFuzz = fuzz,
            samplingNoise = samplingNoise,
            observational = observational,
            rng = thisRng
          )
        }
      case Bags =>
        numBags => Iterator.tabulate(numTrials) { _ =>
          val thisRng = new Random(rng.nextLong())
          runTrial(
            function = function,
            numTrain = numTrain,
            numTest = numTest,
            numCols = numCols,
            numBags = numBags.toInt,
            rhoTrain = rhoTrain,
            quadraticCorrelationFuzz = quadraticCorrelationFuzz,
            samplingNoise = samplingNoise,
            observational = observational,
            rng = thisRng
          )
        }
      case NumTraining =>
        numTraining => Iterator.tabulate(numTrials) { _ =>
          val thisRng = new Random(rng.nextLong())
          runTrial(
            function = function,
            numTrain = numTraining.toInt,
            numTest = numTest,
            numCols = numCols,
            numBags = numTraining.toInt,
            rhoTrain = rhoTrain,
            quadraticCorrelationFuzz = quadraticCorrelationFuzz,
            samplingNoise = samplingNoise,
            observational = observational,
            rng = thisRng
          )
        }
    }
    Merit.plotMeritScan(
      parameterName = variedParameter.name,
      parameterValues = parameterValues,
      merits = merits,
      rng = new Random(0L) // this is irrelevant unless we were to compute Uncertainty Correlation
    )(pvaBuilder)
  }

  def runTrial(
                function: Seq[Double] => Double,
                numTrain: Int,
                numTest: Int,
                numCols: Int,
                numBags: Int,
                rhoTrain: Double,
                quadraticCorrelationFuzz: Double,
                samplingNoise: Double,
                observational: Boolean,
                rng: Random
              ): (PredictionResult[Seq[Any]], Seq[Seq[Any]]) = {
    val dataGenSeed = rng.nextLong()
    val dataNoiseSeed = new Random(rng.nextLong())
    val trainRng = new Random(rng.nextLong())
    val bagSeed = rng.nextLong()
    val linearCorrelationRng = new Random(rng.nextLong())
    val quadraticCorrelationRng = new Random(rng.nextLong())

    val fullData: Seq[(Vector[Double], Double)] = TestUtils.generateTrainingData(
      numTrain + numTest,
      numCols,
      noise = 0.0, // Add noise later, after computing covariate labels
      function = function,
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
      inputs.take(numTrain),
      Seq(realLabelNoise.take(numTrain), linearLabelNoise.take(numTrain), quadraticLabelNoise.take(numTrain))
    ).getModel()

    val predictionResult = RF.transform(inputs.drop(numTrain))
    val trueLabels = Seq(
      realLabel.drop(numTrain),
      linearLabel.drop(numTrain),
      quadraticLabel.drop(numTrain)
    ).transpose
    val observedLabels = Seq(
      realLabelNoise.drop(numTrain),
      linearLabelNoise.drop(numTrain),
      quadraticLabelNoise.drop(numTrain)
    ).transpose
    val actualLabels = if (observational) observedLabels else trueLabels
    (predictionResult, actualLabels)
  }

  def makeQuadraticCorrelatedData(X: Seq[Double], fuzz: Double = 0.0, rng: Random = new Random()): Seq[Double] = {
    require(fuzz >= 0.0)
    val mu = X.sum / X.size
    X.map(x => math.pow(x - mu, 2.0) + fuzz * rng.nextGaussian())
  }

}
