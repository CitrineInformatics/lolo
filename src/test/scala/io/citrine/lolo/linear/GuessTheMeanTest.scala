package io.citrine.lolo.linear

import io.citrine.lolo.api.TrainingRow
import org.junit.Test

class GuessTheMeanTest {

  val learner = GuessTheMeanLearner()

  /** GuessTheMeanLearner should compute the weighted mean of training data. */
  @Test
  def testMean(): Unit = {
    val trainingData = Seq(
      TrainingRow[Double](Vector(), label = 0.0, weight = 1.0),
      TrainingRow[Double](Vector(), label = 6.0, weight = 2.0),
      TrainingRow[Double](Vector(), label = 10.0, weight = 0.0)
    )
    val expectedMean = 4.0 // (0.0 * 1.0 + 6.0 * 2.0 + 10.0 * 0.0) / (1.0 + 2.0 + 0.0)
    val model = learner.train(trainingData).model
    assert(
      model.value == expectedMean,
      s"weighted mean not calculated correctly, expected $expectedMean, got ${model.value}"
    )
  }

  /** If the training values are identical then the mean should be identical (no roundoff error). */
  @Test
  def testIdenticalTrainingValues(): Unit = {
    val label = -0.9248073518671174
    val trainingData = Seq.fill(3)(TrainingRow[Double](Vector(), label = label, weight = 1.0))
    val model = learner.train(trainingData).model
    assert(
      model.value == label,
      s"GuessTheMean learner experiences roundoff error, expected $label, got ${model.value}"
    )
  }
}
