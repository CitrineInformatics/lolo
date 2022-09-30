package io.citrine.lolo.util

import io.citrine.random.Random

/** Tool used to generate Random() instances in LoloPy. */
object LoloPyRandom {

  def getRng(seed: Int): Random = Random(seed)

  def getRng(): Random = Random()

}
