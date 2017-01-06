package io.citrine.lolo.stats.metrics

import org.junit.Test

import scala.util.Random

/**
  * Created by maxhutch on 12/28/16.
  */
@Test
class ClassificationMetricsTest {

  /**
    * Test that the metric works on a sparse confusion matrix
    */
  def testSparse(): Unit = {
    val N = 512
    /* Make random predictions */
    val pva = Seq.tabulate(N){i =>
      (Vector(0.0), Random.nextInt(N).toString, Random.nextInt(N).toString)
    }
    val loss = ClassificationMetrics.f1scores(pva)
    /* The loss should be in (0.0, 1.0) */
    assert(!loss.isNaN)
    assert(loss > 0.0)
    assert(loss < 1.0)
  }

}
