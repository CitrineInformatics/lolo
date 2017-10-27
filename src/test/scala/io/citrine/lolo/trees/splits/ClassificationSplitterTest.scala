package io.citrine.lolo.trees.splits

import io.citrine.lolo.TestUtils
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.theta.Stopwatch


/**
  * Hold some performance profiling routines
  */
class ClassificationSplitterTest {

  /**
    * Evaluate the split finding performance on large and small datasets
    */
  def testSplitterPerformance(): Unit = {
    val timeLarge = Stopwatch.time({
      ClassificationSplitter.getBestSplit(ClassificationSplitterTest.encodedData, 12, 1)
    })
    println(timeLarge)

    val timeSmall = Stopwatch.time({
      (0 until ClassificationSplitterTest.nRow / ClassificationSplitterTest.nSubset).foreach { i =>
        ClassificationSplitter.getBestSplit(ClassificationSplitterTest.subset, 12, 1)
      }
    })
    println(timeSmall)
  }

}

/**
  * Setup the test data
  */
object ClassificationSplitterTest {
  val nRow = 4096
  val nLabel = 4096
  val nSubset = 4

  val testData = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(nRow, 12),
    responseBins = Some(nLabel)
  )

  val encoder = CategoricalEncoder.buildEncoder(testData.map(_._2))
  val encodedData = testData.map{case (f, l) =>
    (f.asInstanceOf[Vector[AnyVal]], encoder.encode(l), 1.0)
  }

  val subset = encodedData.take(nSubset)

  /**
    * Run the tests
    * @param args foo
    */
  def main(args: Array[String]): Unit = {
    new ClassificationSplitterTest().testSplitterPerformance()
  }
}
