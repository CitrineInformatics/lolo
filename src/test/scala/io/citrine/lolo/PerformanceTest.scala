package io.citrine.lolo

import io.citrine.lolo.DataGenerator.TrainingData
import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.bags.{Bagger, ClassificationBagger, MultiTaskBagger, RegressionBagger}
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
      trainingData: Seq[TrainingRow[T]]
  ): (Double, Double) = {
    val inputs = trainingData.map(_.inputs)
    val timeTraining = Stopwatch.time(
      {
        bagger.train(trainingData).model
      },
      benchmark = "None",
      minRun = 4,
      targetError = 0.1,
      maxRun = 32
    )
    val model = bagger.train(trainingData).model

    val timePredicting = Stopwatch.time(
      {
        model.transform(inputs).uncertainty()
      },
      benchmark = "None",
      minRun = 4,
      targetError = 0.1,
      maxRun = 32
    )

    (timeTraining, timePredicting)
  }

  def timeRegressor(
      trainingData: Seq[TrainingRow[Double]],
      n: Int,
      k: Int,
      b: Int
  ): (Double, Double) = {
    val data = trainingData.map(_.mapInputs(inputs => inputs.take(k))).take(n)
    val baseLearner = RegressionTreeLearner(numFeatures = k / 4)
    val bagger = RegressionBagger(baseLearner, numBags = b)
    timedTest(bagger, data)
  }

  def timeClassifier[T](
      trainingData: Seq[TrainingRow[T]],
      n: Int,
      k: Int,
      b: Int
  ): (Double, Double) = {
    val data = trainingData.map(_.mapInputs(inputs => inputs.take(k))).take(n)
    val baseLearner = ClassificationTreeLearner(numFeatures = k / 4)
    val bagger = ClassificationBagger(baseLearner, numBags = b)
    timedTest(bagger, data)
  }

  // @Test
  def testScaling(): Unit = {
    val quiet: Boolean = true
    val Ns = Seq(512, 1024, 2048)
    val Ks = Seq(8, 16, 32)
    val Bs = Seq(1024, 2048, 4096)
    if (!quiet) println(f"${"Train"}%10s, ${"Apply"}%10s, ${"N"}%6s, ${"K"}%6s, ${"B"}%6s")
    timeRegressor(regressionData.data, Ns.head, Ks.head, Bs.head)
    val (bTrain, bApply) = Bs.map(b => timeRegressor(regressionData.data, Ns.head, Ks.head, b)).unzip
    val (kTrain, kApply) =
      (bTrain.zip(bApply).take(1) ++ Ks.tail.map(k => timeRegressor(regressionData.data, Ns.head, k, Bs.head))).unzip
    val (nTrain, nApply) =
      (bTrain.zip(bApply).take(1) ++ Ns.tail.map(n => timeRegressor(regressionData.data, n, Ks.head, Bs.head))).unzip

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
    val (nominalTrain, nominalPredict) = timeClassifier(classificationData.data, 1024, 32, 1024)
    println(nominalTrain, nominalPredict)
  }

  def testMultitaskOverhead(N: Int): (Double, Double) = {
    val realRows = regressionData.data.take(N)
    val avgReal = realRows.map(_.label).sum / realRows.length

    val catLabels = realRows.map(_.label > avgReal)
    val catRows = realRows.zip(catLabels).map { case (row, cat) => row.withLabel(cat) }

    val multiRows = realRows.zip(catRows).map {
      case (real, cat) => real.withLabel(Vector(real.label, cat.label))
    }

    val trainSingle: Double = Stopwatch.time(
      {
        RegressionBagger(RegressionTreeLearner()).train(realRows).loss
        ClassificationBagger(ClassificationTreeLearner()).train(catRows).loss
      },
      minRun = 1,
      maxRun = 1
    )

    val trainMulti: Double = Stopwatch.time(
      {
        MultiTaskBagger(MultiTaskTreeLearner()).train(multiRows, rng = rng).loss
      },
      minRun = 1,
      maxRun = 1
    )

    (trainMulti, trainSingle)
  }

  val regressionData: TrainingData[Double] = DataGenerator.generate(2048, 37, rng = rng)
  val classificationData: TrainingData[String] = regressionData.withBinnedLabels(bins = 8)
}

object PerformanceTest {
  def main(argv: Array[String]): Unit = {
    new PerformanceTest().testAbsolute()
  }
}
