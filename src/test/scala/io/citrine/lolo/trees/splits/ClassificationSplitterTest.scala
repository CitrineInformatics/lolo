package io.citrine.lolo.trees.splits

import io.citrine.lolo.TestUtils
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.theta.Stopwatch
import org.junit.Test


/**
  * Created by maxhutch on 12/1/16.
  */
class ClassificationSplitterTest {

  def testSplitterPerformance(): Unit = {
    val timeLarge = Stopwatch.time({
      ClassificationSplitter.getBestSplit(ClassificationSplitterTest.encodedData, 12, 1)
    })
    println(timeLarge)

    val timeSmall = Stopwatch.time({
      (0 until 256).foreach { i =>
        ClassificationSplitter.getBestSplit(ClassificationSplitterTest.subset, 12, 1)
      }
    })
    println(timeSmall)
  }


}

object ClassificationSplitterTest {
  val testData = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(1024, 12),
    responseBins = Some(256)
  )
  val encoder = CategoricalEncoder.buildEncoder(testData.map(_._2))
  val encodedData = testData.map{case (f, l) =>
    (f.asInstanceOf[Vector[AnyVal]], encoder.encode(l), 1.0)
  }

  val subset = encodedData.take(4)

  def main(args: Array[String]): Unit = {
    new ClassificationSplitterTest().testSplitterPerformance()
  }
}
