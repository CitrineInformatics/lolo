package io.citrine.lolo

import io.citrine.lolo.bags.{Bagger, MultiTaskBagger}
import io.citrine.lolo.trees.classification.ClassificationTreeLearner
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.theta.Stopwatch
import org.junit.Test

/**
  * Created by maxhutch on 12/29/16.
  */
@Test
class PerformanceTest {

  /**
    * Time training and application of models
    * @param trainingData which is used both to train and then later apply the models to
    * @param n number of training rows to take
    * @param k number of features to consider per split
    * @param b number of trees in the forest
    * @param quiet whether to print messages to the screen
    * @return the training and application time, in seconds
    */
  def timedTest(trainingData: Seq[(Vector[Any], Any)], n: Int, k: Int, b: Int, quiet: Boolean = true): (Double, Double) = {
    val data = trainingData.map(p => (p._1.take(k), p._2)).take(n)
    val inputs = data.map(_._1)
    val DTLearner = if (trainingData.head._2.isInstanceOf[Double]) {
      new RegressionTreeLearner(numFeatures = k / 4)
    } else {
      new ClassificationTreeLearner(numFeatures = k / 4)
    }
    val baggedLearner = new Bagger(DTLearner, numBags = b)

    val timeTraining = Stopwatch.time({baggedLearner.train(data).getModel()}, benchmark = "None", minRun = 4, targetError = 0.1, maxRun = 32)
    val model = baggedLearner.train(data).getModel()

    val timePredicting = Stopwatch.time({model.transform(inputs).getUncertainty()}, benchmark = "None", minRun = 4, targetError = 0.1, maxRun = 32)

    if (!quiet) println(f"${timeTraining}%10.4f, ${timePredicting}%10.4f, ${n}%6d, ${k}%6d, ${b}%6d")
    (timeTraining, timePredicting)
  }

  // @Test
  def testScaling(): Unit = {
    val quiet: Boolean = true
    val Ns = Seq(512, 1024, 2048)
    val Ks = Seq(8, 16, 32)
    val Bs = Seq(1024, 2048, 4096)
    if (!quiet) println(f"${"Train"}%10s, ${"Apply"}%10s, ${"N"}%6s, ${"K"}%6s, ${"B"}%6s")
    timedTest(trainingData, Ns.head, Ks.head, Bs.head, true)
    val (bTrain, bApply) = Bs.map(b => timedTest(trainingData, Ns.head, Ks.head, b, quiet)).unzip
    val (kTrain, kApply) = (bTrain.zip(bApply).take(1) ++ Ks.tail.map(k => timedTest(trainingData, Ns.head, k, Bs.head, quiet))).unzip
    val (nTrain, nApply) = (bTrain.zip(bApply).take(1) ++ Ns.tail.map(n => timedTest(trainingData, n, Ks.head, Bs.head, quiet))).unzip

    val bTrainScale = (1 until bTrain.size).map(i => bTrain(i)/bTrain(i-1))
    val nTrainScale = (1 until nTrain.size).map(i => nTrain(i)/nTrain(i-1))
    val kTrainScale = (1 until kTrain.size).map(i => kTrain(i)/kTrain(i-1))
    val bApplyScale = (1 until bApply.size).map(i => bApply(i)/bApply(i-1))
    val nApplyScale = (1 until nApply.size).map(i => nApply(i)/nApply(i-1))
    val kApplyScale = (1 until kApply.size).map(i => kApply(i)/kApply(i-1))

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
    // val (nominalTrain, nominalPredict) = timedTest(trainingData, 1024, 32, 1024)
    val (nominalTrain, nominalPredict) = timedTest(classificationData, 1024, 32, 1024)
    println(nominalTrain, nominalPredict)
    // assert(nominalTrain < 8.0, s"Expected nominal train to have theta < 8.0 but was ${nominalTrain}")
    // assert(nominalPredict < 7.0, s"Expected nominal transform to have theta < 7.0 but was ${nominalPredict}")
  }

  def testMultitaskOverhead(N: Int): (Double, Double) = {
    val subset = trainingData.take(N)
    val inputs = subset.map(_._1)
    val realLabels = subset.map(_._2)
    val catLabels = realLabels.map(_ > realLabels.sum / realLabels.size)

    val trainSingle: Double = Stopwatch.time({
      new Bagger(new RegressionTreeLearner()).train(inputs.zip(realLabels)).getLoss()
      new Bagger(new ClassificationTreeLearner()).train(inputs.zip(catLabels)).getLoss()
    }, minRun = 1, maxRun = 1)

    val trainMulti: Double = Stopwatch.time({
      new MultiTaskBagger(new MultiTaskTreeLearner()).train(inputs, Seq(realLabels, catLabels)).map(_.getLoss())
    }, minRun = 1, maxRun = 1)

    (trainMulti, trainSingle)
  }

  val trainingData = TestUtils.generateTrainingData(2048, 37)
  val classificationData = TestUtils.binTrainingData(trainingData, responseBins = Some(8))
}

object PerformanceTest {
  def main(argv: Array[String]): Unit = {
    new PerformanceTest().testAbsolute()

  }
}
