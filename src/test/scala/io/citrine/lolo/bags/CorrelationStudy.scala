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

import java.io.{BufferedWriter, File, FileWriter}
import scala.util.Random

/** A description of the relationship between the two variables for which the correlation estimates are to be studied. */
sealed trait TestProblem {
  def name: String
}
/** The two variables are generated to have some fixed Pearson correlation coefficient */
case object Linear extends TestProblem {
  def name = "linear correlation"
}
/** Two variables are generated with a quadratic relationship between them, and then some normally-distributed noise
  * is applied to one variable. The larger the level of noise, the less correlated the two variables are.
  */
case object Quadratic extends TestProblem {
  def name = "quadratic relationship"
}

/** Metric used to evaluate the accuracy of the estimated covariance matrix. */
sealed trait Metric {
  def name: String
}
case object NLPD extends Metric {
  def name = "negative log probability density"
}
case object StdConfidence extends Metric {
  def name = "1-sigma standard confidence"
}

/** Underlying function from which to draw training and test data. */
sealed trait TrueFunction {
  def function: Seq[Double] => Double
  val numCols: Int
  def name: String
}
case class FriedmanSilvermanFunction(numCols: Int) extends TrueFunction {
  def name = s"Friedman-Silverman $numCols columns"
  def function: Seq[Double] => Double = Friedman.friedmanSilverman
}
case class FriedmanGrosseSilvermanFunction(numCols: Int) extends TrueFunction {
  def name = s"Friedman-Grosse-Silverman $numCols columns"
  def function: Seq[Double] => Double = Friedman.friedmanGrosseSilverman
}

/** Parameter to vary as part of a scan. */
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
/** If the number of bags is not being varied, then the default is to set numBags = numTrain */
case object Bags extends VariedParameter {
  def name: String = "number of bags"
}
case object NumTraining extends VariedParameter {
  def name: String = "number of training rows"
}

object CorrelationStudy {

  def main(args: Array[String]): Unit = {
    val mainRng = new Random(52109317L)

    runTrialsAndSave(
      fname = "./test",
      metric = NLPD,
      variedParameter = TrainRho,
      parameterValues = Seq(0.0, 0.25, 0.50, 0.75, 0.9, 0.99),
      testProblem = Linear,
      function = FriedmanSilvermanFunction(numCols = 12),
      numTrials = 10,
      numTrain = 128,
      numTest = 128,
      observational = true,
      samplingNoise = 0.0,
      rhoTrain = 0.0,
      quadraticCorrelationFuzz = 0.0,
      rng = mainRng
    )
  }

  /**
    * Scan some figure of merit over some parameter and save the results both as a chart and a table.
    *
    * @param fname file name to save to
    * @param metric figure of merit to calculate over the predictions
    * @param variedParameter the parameter to vary as part of the scan
    * @param parameterValues values that the varied parameter takes on
    * @param testProblem relationship bewteen the two variables that are being tested
    * @param function function to generate training and test data
    * @param numTrials number of independent trials to run and combine the results of
    * @param numTrain number of training data points
    * @param numTest number of test data points
    * @param observational whether to compare the prediction to the observed value (with noise) or the true value
    * @param samplingNoise level of normally-distributed random noise to apply to the training and test data
    * @param rhoTrain linear correlation coefficient bewteen label 0 and label 1
    * @param quadraticCorrelationFuzz level of normally-distributed random values to apply to the second label
    *                                 in the quadratic pair. Note that this value modifies the true underlying data,
    *                                 which is separate from applying sampling noise.
    * @param rng random number generator, to make it mostly reproducible. Because the trees are trained in parallel
    *            and we do not have splittable random numbers, it is impossible to make the results completely reproducible.
    *            But the data generation, application of noise, and Poisson draws for bagging are reproducible.
    */
  def runTrialsAndSave(
                        fname: String,
                        metric: Metric,
                        variedParameter: VariedParameter,
                        parameterValues: Seq[Double],
                        testProblem: TestProblem,
                        function: TrueFunction,
                        numTrials: Int,
                        numTrain: Int,
                        numTest: Int,
                        observational: Boolean,
                        samplingNoise: Double,
                        rhoTrain: Double,
                        quadraticCorrelationFuzz: Double,
                        rng: Random
                      ): Unit = {
    val chart = makeChart(
      variedParameter = variedParameter,
      metric = metric,
      parameterValues = parameterValues,
      testProblem = testProblem,
      function = function,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      observational = observational,
      samplingNoise = samplingNoise,
      rhoTrain = rhoTrain,
      quadraticCorrelationFuzz = quadraticCorrelationFuzz,
      rng = rng
    )

    saveChart(chart, fname)
    saveRawData(
      chart = chart,
      fname = fname,
      metric = metric,
      variedParameter = variedParameter,
      parameterValues = parameterValues,
      testProblem = testProblem,
      function = function,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      observational = observational,
      samplingNoise = samplingNoise,
      rhoTrain = rhoTrain,
      quadraticCorrelationFuzz = quadraticCorrelationFuzz
    )
  }

  def makeChart(
                 variedParameter: VariedParameter,
                 metric: Metric,
                 parameterValues: Seq[Double],
                 testProblem: TestProblem,
                 function: TrueFunction,
                 numTrials: Int,
                 numTrain: Int,
                 numTest: Int,
                 observational: Boolean,
                 samplingNoise: Double,
                 rhoTrain: Double,
                 quadraticCorrelationFuzz: Double,
                 rng: Random
               ): XYChart = {
    // the linear problem compares label 0 to label 1, and the quadratic problem compares label 0 to label 2
    val index: Int = testProblem match {
      case Linear => 1
      case Quadratic => 2
    }
    // Select the figures of merit based on the desired metric
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
    // There's some unfortunate boiler plate here. plotMeritScan takes a function that goes from a Double parameter value
    // to an iterator of what is essentially predicted-vs-actuals. This always uses the runTrial method, but for each
    // case we need to separately pick where the parameter gets plugged in.
    val pvaBuilder: Double => Iterator[(PredictionResult[Seq[Any]], Seq[Seq[Any]])] = variedParameter match {
      case TrainRho =>
        rho => Iterator.tabulate(numTrials) { _ =>
          val thisRng = new Random(rng.nextLong())
          runTrial(
            function = function,
            numTrain = numTrain,
            numTest = numTest,
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
      rng = new Random(0L) // this is irrelevant unless we were computing Uncertainty Correlation
    )(pvaBuilder)
  }

  def runTrial(
                function: TrueFunction,
                numTrain: Int,
                numTest: Int,
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
      function.numCols,
      noise = 0.0, // Add noise later, after computing covariate labels
      function = function.function,
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

  /** Save chart as a png */
  def saveChart(chart: XYChart, fname: String): Unit = BitmapEncoder.saveBitmap(chart, fname + ".png", BitmapFormat.PNG)

  /** Save the underlying data for a chart as a csv. */
  def saveRawData(
                   chart: XYChart,
                   fname: String,
                   metric: Metric,
                   variedParameter: VariedParameter,
                   parameterValues: Seq[Double],
                   testProblem: TestProblem,
                   function: TrueFunction,
                   numTrials: Int,
                   numTrain: Int,
                   numTest: Int,
                   observational: Boolean,
                   samplingNoise: Double,
                   rhoTrain: Double,
                   quadraticCorrelationFuzz: Double,
                 ): Unit = {
    val path = fname + ".csv"
    val headers = Seq(
      "correlation method", "metric", "mean value", "std error of value",
      "function", "trials", "train", "test", "bags", "observational",
      "sample noise", "linear rho", "quadratic fuzz"
    )
    writeCSVRow(path, headers, append = false)
    var thisNumTrain = numTrain
    var thisNumBags = numTrain
    var thisSamplingNoise = samplingNoise
    var thisRhoTrain = rhoTrain
    var thisQuadadraticFuzz = quadraticCorrelationFuzz

    parameterValues.zipWithIndex.foreach { case (parameterValue, index) =>
      // assign parameterValue to the appropriate variable
      variedParameter match {
        case TrainRho => thisRhoTrain = parameterValue
        case TrainQuadraticFuzz => thisQuadadraticFuzz = parameterValue
        case Noise => thisSamplingNoise = parameterValue
        case Bags => thisNumBags = parameterValue.toInt
        case NumTraining =>
          thisNumTrain = parameterValue.toInt
          thisNumBags = parameterValue.toInt
      }
      // For each metric, pull out the result for this parameter value and write a row of the CSV
      chart.getSeriesMap.forEach { case (key, series) =>
        val y = series.getYData.apply(index)
        val yErr = series.getExtraValues.apply(index)
        // the order of variables here must match the order of the headers, above
        val rowData = Seq(
          key, metric.name, y, yErr,
          function.name, numTrials, thisNumTrain, numTest, thisNumBags, observational,
          thisSamplingNoise, thisRhoTrain, thisQuadadraticFuzz
        )
        writeCSVRow(path, rowData, append = true)
      }
    }

  }

  /**
    * Write a sequence of values as a row of a CSV
    *
    * @param path path to CSV file
    * @param entries values to be written
    * @param append whether or not to append if file exists already
    */
  def writeCSVRow(path: String, entries: Seq[Any], append: Boolean): Unit = {
    val bw = new BufferedWriter(new FileWriter(new File(path), append))
    bw.write(entries.map(_.toString).mkString(start = "", sep = ",", end = "\n"))
    bw.close()
  }

}
