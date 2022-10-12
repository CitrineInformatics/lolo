package io.citrine.lolo

import breeze.stats.variance
import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.random.Random

object DataGenerator {

  /** Sequence of inputs and labels generated for training a learner. */
  trait TrainingData[+T] { def data: Seq[(Vector[Any], T)] }

  case class ClassificationData(data: Seq[(Vector[Any], String)]) extends TrainingData[String]

  case class RegressionData(data: Seq[(Vector[Any], Double)]) extends TrainingData[Double] {
    def withBinnedInputs(bins: Seq[(Int, Int)]): RegressionData = {
      val (baseInputs, baseLabels) = data.unzip
      val indexToBins = bins.toMap
      val transposedInputs = baseInputs.transpose
      val binnedInputs = Vector
        .tabulate(transposedInputs.length) { index =>
          val indexData = transposedInputs(index).asInstanceOf[Seq[Double]]
          indexToBins
            .get(index)
            .map { nBins => binData(indexData, nBins) }
            .getOrElse(indexData)
        }
        .transpose
      RegressionData(binnedInputs.zip(baseLabels))
    }

    def withBinnedLabels(bins: Int): ClassificationData = {
      val (baseInputs, baseLabels) = data.unzip
      val binnedLabels = binData(baseLabels, bins)
      ClassificationData(baseInputs.zip(binnedLabels))
    }
  }

  private def binData(continuousData: Seq[Double], nBins: Int): Seq[String] = {
    val min = continuousData.min
    val max = continuousData.max
    continuousData.map(x => math.round(x * nBins / (max - min)).toString)
  }

  def generate(
      rows: Int,
      cols: Int,
      function: Seq[Double] => Double = Friedman.friedmanGrosseSilverman,
      xscale: Double = 1.0,
      xoff: Double = 0.0,
      noise: Double = 0.0,
      rng: Random = Random()
  ): RegressionData = {
    val data = Vector.fill(rows) {
      val input = Vector.fill(cols)(xscale * rng.nextDouble() + xoff)
      (input, function(input) + noise * rng.nextGaussian())
    }
    RegressionData(data)
  }

  def iterate(
      cols: Int,
      function: Seq[Double] => Double = Friedman.friedmanGrosseSilverman,
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

  /**
    * Given a univariate data set, construct a corresponding data set with the desired Pearson correlation coefficient.
    * The procedure is to generate new data randomly, calculate the residuals of a linear regression,
    * and then create a suitable linear combination of X and the residuals.
    * Source: https://stats.stackexchange.com/a/313138
    *
    * @param X   sequence of values
    * @param rho desired Pearson correlation coefficient
    * @param rng random number generator
    * @return sequence of values that have desired correlation with X
    */
  def makeLinearlyCorrelatedData(X: Seq[Double], rho: Double, rng: Random = Random()): Seq[Double] = {
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
}
