package io.citrine.lolo.trees.splits

import scala.collection.BitSet

/**
  * Splits are used by decision trees to partition the input space
  */
sealed trait Split extends Serializable {

  /**
    * Take the left branch in the binary split?
    *
    * @param input vector of any type
    * @return true if input takes the left split
    */
  def turnLeft(input: Vector[Any]): Boolean

  /**
    * Get the index of the input vector that is used to pick this split
    *
    * @return index of the input vector used by this split
    */
  def index: Int
}

/**
  * If no split was found
  */
case class NoSplit() extends Split {

  /**
    * Take the left branch in the binary split?
    *
    * @param input vector of any type
    * @return true if input takes the left split
    */
  override def turnLeft(input: Vector[Any]): Boolean = false

  /**
    * Get the index of the input vector that is used to pick this split
    *
    * @return index of the input vector used by this split
    */
  override def index: Int = -1
}

/**
  * Split based on a real value in the index position
  *
  * @param index position of the real value to inspect
  * @param pivot value at or below which to take the left split
  */
case class RealSplit(index: Int, pivot: Double) extends Split {

  /**
    * If the value is at or less than the pivot, turn left
    *
    * @param input vector of any type
    * @return true if input takes the left split
    */
  override def turnLeft(input: Vector[Any]): Boolean = {
    if (pivot.isNaN) {
      // the normal sort order for scala puts NaN >> any other double
      // so if the pivot is NaN and the value is not, then it is 'less than' the pivot
      !input(index).asInstanceOf[Double].isNaN
    } else {
      input(index).asInstanceOf[Double] <= pivot
    }
  }

  /**
    * Pretty print
    *
    * @return debug string
    */
  override def toString: String = s"Split index $index @ $pivot"
}

/**
  * Split based on inclusion in a set
  *
  * @param index      of the categorical feature
  * @param includeSet set of values that turn left
  */
case class CategoricalSplit(index: Int, includeSet: BitSet) extends Split {

  /**
    * If the value at the index position is in the set, turn left
    *
    * @param input vector of any type
    * @return true if input takes the left split
    */
  override def turnLeft(input: Vector[Any]): Boolean = {
    includeSet.contains(input(index).asInstanceOf[Char].toInt)
  }
}
