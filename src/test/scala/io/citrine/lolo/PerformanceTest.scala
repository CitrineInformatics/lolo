package io.citrine.lolo

import io.citrine.lolo.bags.Bagger
import io.citrine.lolo.trees.RegressionTreeLearner
import org.junit.Test
import org.junit.experimental.categories.Category

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
  def timedTest(trainingData: Seq[(Vector[Any], Double)], n: Int, k: Int, b: Int, quiet: Boolean = true): (Double, Double) = {
    val data = trainingData.map(p => (p._1.take(k), p._2)).take(n)
    val inputs = data.map(_._1)
    val DTLearner = new RegressionTreeLearner(numFeatures = k / 4)
    val baggedLearner = new Bagger(DTLearner, numBags = b)
    val nIter = 4

    baggedLearner.train(data).getModel()
    val startTraining = System.nanoTime()
    val model = baggedLearner.train(data).getModel()
    (0 until nIter - 1).foreach {i =>
      val tmp = baggedLearner.train(data).getModel()
    }
    val timeTraining = (System.nanoTime() - startTraining) / 1.0e9 / nIter

    model.transform(inputs).getUncertainty()
    val startPredicting = System.nanoTime()
    val res = model.transform(inputs).getUncertainty()
    (0 until nIter - 1).foreach {i =>
      val tmp = model.transform(inputs).getUncertainty()
    }
    val timePredicting = (System.nanoTime() - startPredicting) / 1.0e9 / nIter

    if (!quiet) println(f"${timeTraining}%10.4f, ${timePredicting}%10.4f, ${n}%6d, ${k}%6d, ${b}%6d")
    (timeTraining, timePredicting)
  }

  @Test
  @Category(Array(classOf[SlowTest]))
  def benchmark(): Unit = {
    val quiet: Boolean = true
    val trainingData = TestUtils.generateTrainingData(47667, 37)
    // val Ns = Seq(8192, 16384, 32768)
    val Ns = Seq(1024, 2048, 4096)
    val Ks = Seq(8, 16, 32)
    val Bs = Seq(512, 1024, 2048)
    if (!quiet) println(f"${"Train"}%10s, ${"Apply"}%10s, ${"N"}%6s, ${"K"}%6s, ${"B"}%6s")
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
    assert(nApplyScale.forall(s => s < Math.sqrt(32.0) && s > Math.sqrt(8.0)), nApplyScale)
  }

}

object PerformanceTest {
  def main(argv: Array[String]): Unit = {
    new PerformanceTest().benchmark()
  }
}
