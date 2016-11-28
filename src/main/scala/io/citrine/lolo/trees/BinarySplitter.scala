package io.citrine.lolo.trees

/**
  * Created by maxhutch on 11/27/16.
  */
abstract trait BinarySplitter {

  /**
    * Split the input into either the left or the right leaf
    * @param input to split on
    * @return true for left, false for right
    */
  def split(input: Any): Boolean
}

abstract trait BinarySplitGenerator {

  /**
    * Look at the values present and generate potential splits
    * @param values to split over
    * @return iterator over binary splitters
    */
  def getSplits(values: Seq[Any]): Iterator[BinarySplitter]
}
