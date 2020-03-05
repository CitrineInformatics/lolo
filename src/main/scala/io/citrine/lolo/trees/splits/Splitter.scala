package io.citrine.lolo.trees.splits

/**
  * Created by maxhutch on 7/5/17.
  */
trait Splitter[T] {
  def getBestSplit(data: Seq[(Vector[AnyVal], T, Double)], numFeatures: Int, minInstances: Int): (Split, Double)
}

object Splitter {
  def isDifferent(x1: Double, x2: Double): Boolean = {
    if (x1 == 0 || x2 == 0) {
      x1 != x2
    } else if (x1 * x2 < 0) {
      true
    } else {
      Math.abs((x1 - x2) / (x1 + x2)) > 1.0e-9
    }
  }
}
