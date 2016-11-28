package io.citrine.lolo.trees

/**
  * Created by maxhutch on 11/27/16.
  */
class RealSplitter(pivot: Double) extends BinarySplitter {

  /**
    * Split the input into either the left or the right leaf
    *
    * @param input to split on
    * @return true for left, false for right
    */
  override def split(input: Any): Boolean = {
    input.asInstanceOf[Double] > pivot
  }
}

class RealSplitIterator(values: Vector[Double]) extends Iterator[RealSplitter] {

  override def hasNext: Boolean = index < values.size - 1

  override def next(): RealSplitter = {
    index = index + 1
    new RealSplitter((sortedValues(index-1) + sortedValues(index)) / 2.0)
  }

  var index: Int = 0
  val sortedValues: Vector[Double] = values.sorted
}

object RealSplitGenerator extends BinarySplitGenerator {
  /**
    * Look at the values present and generate potential splits
    *
    * @param values to split over
    * @return iterator over binary splitters
    */
  override def getSplits(values: Seq[Any]): Iterator[BinarySplitter] = {
    new RealSplitIterator(values.asInstanceOf[Vector[Double]])
  }
}
