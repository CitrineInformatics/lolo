package io.citrine.lolo.trees.splits

import io.citrine.lolo.{SeedRandomMixIn, TestUtils}
import io.citrine.lolo.encoders.CategoricalEncoder
import io.citrine.theta.Stopwatch

/**
  * Hold some performance profiling routines.
  *
  * NOTE: performance on travis isn't sufficiently stable to run these continouously.
  * For now, they are intended as developer tools for collecting performance signal.
  */
class ClassificationSplitterTest extends SeedRandomMixIn {

  val nRow = 4096
  val nLabel = 4096
  val nSubset = 4

  val testData: Seq[(Vector[Any], Any)] = TestUtils.binTrainingData(
    TestUtils.generateTrainingData(nRow, 12, rng = rng),
    responseBins = Some(nLabel)
  )

  val encoder: CategoricalEncoder[Any] = CategoricalEncoder.buildEncoder(testData.map(_._2))
  val encodedData: Seq[(Vector[AnyVal], Char, Double)] = testData.map {
    case (f, l) =>
      (f.asInstanceOf[Vector[AnyVal]], encoder.encode(l), 1.0)
  }

  val subset: Seq[(Vector[AnyVal], Char, Double)] = encodedData.take(nSubset)

  /**
    * Evaluate the split finding performance on large and small datasets
    */
  def testSplitterPerformance(): Unit = {
    val timeLarge = Stopwatch.time {
      ClassificationSplitter().getBestSplit(encodedData, 12, 1)
    }

    val timeSmall = Stopwatch.time {
      (0 until nRow / nSubset).foreach { _ =>
        ClassificationSplitter().getBestSplit(subset, 12, 1)
      }
    }
    println(s"The normalized cost of splitting is ${timeLarge} and ${timeSmall} for large and small, resp.")
  }
}

/**
  * Setup the test data
  */
object ClassificationSplitterTest {

  /**
    * Run the tests
    *
    * @param args foo
    */
  def main(args: Array[String]): Unit = {
    new ClassificationSplitterTest().testSplitterPerformance()
  }
}
