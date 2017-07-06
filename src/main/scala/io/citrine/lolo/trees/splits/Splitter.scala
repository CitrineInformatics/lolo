package io.citrine.lolo.trees.splits

/**
  * Created by maxhutch on 7/5/17.
  */
trait Splitter[T] {
  def getBestSplit(data: Seq[(Vector[AnyVal], T, Double)], numFeatures: Int, minInstances: Int): (Split, Double)
}
