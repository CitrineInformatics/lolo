package io.citrine.lolo

import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.StatsUtils.variance
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.random.Random
import org.junit.Before

import scala.util.Try

/**
  * Created by maxhutch on 11/28/16.
  */
object TestUtils {

  def readCsv(name: String): Seq[Vector[Any]] = {
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    val res = bs
      .getLines()
      .flatMap { line =>
        Try(
          line
            .split(",")
            .map(_.trim)
            .map { token =>
              try {
                token.toDouble
              } catch {
                case _: Throwable if token == "NaN" => Double.NaN
                case _: Throwable if token.nonEmpty => token
              }
            }
            .toVector
        ).toOption
      }
      .toVector
    bs.close()
    res
  }

  def generateTrainingData(
      rows: Int,
      cols: Int,
      function: (Seq[Double] => Double) = Friedman.friedmanGrosseSilverman,
      xscale: Double = 1.0,
      xoff: Double = 0.0,
      noise: Double = 0.0,
      rng: Random = Random()
  ): Vector[(Vector[Double], Double)] = {
    Vector.fill(rows) {
      val input = Vector.fill(cols)(xscale * rng.nextDouble() + xoff)
      (input, function(input) + noise * rng.nextGaussian())
    }
  }

  /** Given a univariate data set, construct a corresponding data set with the desired Pearson correlation coefficient.
    * The procedure is to generate new data randomly, calculate the residuals of a linear regression,
    * and then create a suitable linear combination of X and the residuals.
    * Source: https://stats.stackexchange.com/a/313138
    *
    * @param X    sequence of values
    * @param rho  desired Pearson correlation coefficient
    * @param rng  random number generator
    * @return     sequence of values that have desired correlation with X
    */
  def makeLinearCorrelatedData(X: Seq[Double], rho: Double, rng: Random = Random()): Seq[Double] = {
    require(rho >= -1.0 && rho <= 1.0, "correlation coefficient must be between -1.0 and 1.0")
    val Y = Seq.fill(X.length)(rng.nextGaussian())
    val linearLearner = LinearRegressionLearner()
    val linearModel = linearLearner.train(X.zip(Y).map { case (x, y) => (Vector(x), y) }).getModel()
    val yPred = linearModel.transform(X.map(Vector(_))).getExpected()
    val residuals = Y.zip(yPred).map { case (actual, predicted) => actual - predicted }
    val stdX = math.sqrt(variance(X))
    val stdResiduals = math.sqrt(variance(residuals))
    X.zip(residuals).map { case (x, residual) => rho * stdResiduals * x + math.sqrt(1 - rho * rho) * stdX * residual }
  }

  def iterateTrainingData(
      cols: Int,
      function: (Seq[Double] => Double) = Friedman.friedmanGrosseSilverman,
      xscale: Double = 1.0,
      xoff: Double = 0.0,
      noise: Double = 0.0,
      rng: Random = Random()
  ): Iterator[(Vector[Double], Double)] = {
    Iterator.continually {
      val input = Vector.fill(cols)(xscale * rng.nextDouble() + xoff)
      (input, function(input) + noise * rng.nextGaussian())
    }
  }

  // TODO(PLA-10433): Better scheme for training data generation/binning alongside TrainingRow classes
  def binTrainingResponses(continuousData: Vector[Double], nBins: Int): Vector[String] = {
    val min = continuousData.min
    val max = continuousData.max
    continuousData.map(x => math.round(x * nBins / (max - min)).toString)
  }

  def binTrainingInputs(
      continuousData: Seq[Vector[Double]],
      bins: Seq[(Int, Int)] = Seq.empty
  ): Seq[Vector[Any]] = {
    val indexToBins = bins.toMap
    val transposedData = continuousData.toVector.transpose
    // Tabulating as a vector to maintain Vector type during transpose
    Vector
      .tabulate(transposedData.length) { index =>
        val indexData = transposedData(index)
        indexToBins
          .get(index)
          .map { nBins => binTrainingResponses(indexData, nBins) }
          .getOrElse(indexData)
      }
      .transpose
  }

  /**
    * Enumerate the cartesian product of items in baseGrids.
    *
    * @param baseGrids a sequence of 1-d mesh specifications, one for each dimension of the output vectors
    * @return a sequence of vectors enumerating the cartesian product of items in baseGrids
    */
  def enumerateGrid(baseGrids: Seq[Seq[Double]]): Seq[Vector[Double]] = {
    if (baseGrids.length == 1) {
      baseGrids.head.map { x => Vector(x) }
    } else {
      baseGrids.head.flatMap { x =>
        enumerateGrid(baseGrids.takeRight(baseGrids.length - 1)).map { n => x +: n }
      }
    }
  }

}

/**
  * Mix-in class to facilitate predictable random number streams.
  */
trait SeedRandomMixIn {
  // Reset random number generator.
  var rng: Random = Random(2348752L)

  @Before
  def initializeRandom(): Unit = {
    // Seeds must also be reset for each test so that incremental tests are as predictable as running the full case.
    rng = Random(2348752L)
  }
}
