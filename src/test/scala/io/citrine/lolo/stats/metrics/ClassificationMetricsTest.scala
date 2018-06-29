package io.citrine.lolo.stats.metrics

import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 12/28/16.
  */
class ClassificationMetricsTest {

  /**
    * Test that the metric works on a sparse confusion matrix
    */
  @Test
  def testSparse(): Unit = {
    val N = 512
    /* Make random predictions */
    val pva = Seq.tabulate(N){i =>
      (Vector(0.0), Random.nextInt(N).toString, i.toString)
    }
    val loss = ClassificationMetrics.f1scores(pva)
    /* The loss should be in (0.0, 1.0) */
    assert(!loss.isNaN, s"Loss was NaN")
    assert(loss >= 0.0, s"Loss was less than 0: ${loss}")
    assert(loss <= 1.0, s"Loss was greater than 1: ${loss}")
  }
}
