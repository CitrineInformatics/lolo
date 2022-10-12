package io.citrine.lolo

import io.citrine.lolo.bags.{Bagger, MultiTaskBagger}
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.theta.Stopwatch

/**
  * Performance tests
  *
  * NOTE: the performance is not sufficiently stable on travis to
  * run these continuously.  Therefore, they've had their @Test annotations
  * removed and are meant to be run locally to collect a performance signal while developing.
  *
  * Created by maxhutch on 12/29/16.
  */
class PerformanceTest extends SeedRandomMixIn {

  private def timedTest[T](
      bagger: Bagger[T],
      trainingData: Seq[(Vector[Any], T)]
  ): (Double, Double) = {
    val inputs = trainingData.map(_._1)
    val timeTraining = Stopwatch.time(
      {
        bagger.train(trainingData).getModel()
      },
      benchmark = "None",
      minRun = 4,
      targetError = 0.1,
      maxRun = 32
    )
    val model = bagger.train(trainingData).getModel()

    val timePredicting = Stopwatch.time(
      {
        model.transform(inputs).getUncertainty()
      },
      benchmark = "None",
      minRun = 4,
      targetError = 0.1,
      maxRun = 32
    )

    (timeTraining, timePredicting)
  }

  def timeRegressor(
      trainingData: Seq[(Vector[Any], Double)],
      n: Int,
      k: Int,
      b: Int
  ): (Double, Double) = {
    val data = trainingData.map(p => (p._1.take(k), p._2)).take(n)
    val baseLearner = RegressionTreeLearner(numFeatures = k / 4)
    val bagger = new Bagger(baseLearner, numBags = b)
    timedTest(bagger, data)
  }

  def timeClassifier[T](
      trainingData: Seq[(Vector[Any], T)],
      n: Int,
      k: Int,
      b: Int
  ): (Double, Double) = {
    val data = trainingData.map(p => (p._1.take(k), p._2)).take(n)
    val baseLearner = ClassificationTreeLearner(numFeatures = k / 4)
    val bagger = new Bagger(baseLearner, numBags = b)
    timedTest(bagger, data)
  }

  // @Test
  def testScaling(): Unit = {
    val quiet: Boolean = true
    val Ns = Seq(512, 1024, 2048)
    val Ks = Seq(8, 16, 32)
    val Bs = Seq(1024, 2048, 4096)
    if (!quiet) println(f"${"Train"}%10s, ${"Apply"}%10s, ${"N"}%6s, ${"K"}%6s, ${"B"}%6s")
    timeRegressor(regressionData, Ns.head, Ks.head, Bs.head)
    val (bTrain, bApply) = Bs.map(b => timeRegressor(regressionData, Ns.head, Ks.head, b)).unzip
    val (kTrain, kApply) =
      (bTrain.zip(bApply).take(1) ++ Ks.tail.map(k => timeRegressor(regressionData, Ns.head, k, Bs.head))).unzip
    val (nTrain, nApply) =
      (bTrain.zip(bApply).take(1) ++ Ns.tail.map(n => timeRegressor(regressionData, n, Ks.head, Bs.head))).unzip

    val bTrainScale = (1 until bTrain.size).map(i => bTrain(i) / bTrain(i - 1))
    val nTrainScale = (1 until nTrain.size).map(i => nTrain(i) / nTrain(i - 1))
    val kTrainScale = (1 until kTrain.size).map(i => kTrain(i) / kTrain(i - 1))
    val bApplyScale = (1 until bApply.size).map(i => bApply(i) / bApply(i - 1))
    val nApplyScale = (1 until nApply.size).map(i => nApply(i) / nApply(i - 1))
    val kApplyScale = (1 until kApply.size).map(i => kApply(i) / kApply(i - 1))

    assert(bTrainScale.forall(s => s < Math.sqrt(8.0) && s > Math.sqrt(2.0)), bTrainScale)
    assert(kTrainScale.forall(s => s < Math.sqrt(8.0) && s > Math.sqrt(0.5)), kTrainScale)
    assert(nTrainScale.forall(s => s < Math.sqrt(8.0) && s > Math.sqrt(2.0)), nTrainScale)

    assert(bApplyScale.forall(s => s < Math.sqrt(8.0) && s > Math.sqrt(2.0)), bApplyScale)
    assert(kApplyScale.forall(s => s < Math.sqrt(2.0) && s > Math.sqrt(0.5)), kApplyScale)
    // Relaxing the lower bound because the blocked matrix-vector products can be super-linear on the margin
    // (because of the cost of loading the matrix into memory)
    assert(nApplyScale.forall(s => s < Math.sqrt(32.0) && s > Math.sqrt(4.0)), nApplyScale)
  }

  /**
    * Test the absolute performance to check for overall regressions
    */
  def testAbsolute(): Unit = {
    val (nominalTrain, nominalPredict) = timeClassifier(classificationData, 1024, 32, 1024)
    println(nominalTrain, nominalPredict)
  }

  def testMultitaskOverhead(N: Int): (Double, Double) = {
    val subset = regressionData.take(N)
    val (inputs, realLabels) = subset.unzip
    val catLabels = realLabels.map(_ > realLabels.sum / realLabels.size)
    val labels = Vector(realLabels, catLabels).transpose

    val trainSingle: Double = Stopwatch.time(
      {
        new Bagger(RegressionTreeLearner()).train(inputs.zip(realLabels)).getLoss()
        new Bagger(ClassificationTreeLearner()).train(inputs.zip(catLabels)).getLoss()
      },
      minRun = 1,
      maxRun = 1
    )

    val trainMulti: Double = Stopwatch.time(
      {
        MultiTaskBagger(MultiTaskTreeLearner()).train(inputs.zip(labels)).getLoss()
      },
      minRun = 1,
      maxRun = 1
    )

    (trainMulti, trainSingle)
  }

  val regressionData: Vector[(Vector[Double], Double)] = TestUtils.generateTrainingData(2048, 37, rng = rng)
  val classificationData: Seq[(Vector[Any], Any)] = {
    val (baseLabels, baseResponses) = regressionData.unzip
    val binnedResponses = TestUtils.binTrainingResponses(baseResponses, nBins = 8)
    baseLabels.zip(binnedResponses)
  }
}

object PerformanceTest {
  def main(argv: Array[String]): Unit = {
    new PerformanceTest().testAbsolute()
  }
}
