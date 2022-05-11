package io.citrine.lolo.bags

import io.citrine.lolo.SeedRandomMixIn
import org.junit.Test

class BaggerUtilTest extends SeedRandomMixIn {

  /** Test that the rescale ratio is tolerant to error/uncertainty close to zero. */
  @Test
  def testRescaleRatioPrecision(): Unit = {
    val almostZeros = Vector.fill(75)((1e-14, 1e-16)) // produces ratio of 1.0 due to approximately equality
    val zeroUncertainty = Vector.fill(25)((1.0 + rng.nextDouble(), 0.0)) // produces ratio of Infinity
    val oobData = almostZeros ++ zeroUncertainty
    val ratio = BaggerUtil.calculateRescaleRatio(oobData)
    assert(ratio == 1.0)
  }
}
