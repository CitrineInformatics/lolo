package io.citrine.lolo.bags

sealed trait BaggedTrainingError extends Throwable {
  def numRows: Int
  def numRequired: Int
}

case class InsufficientTrainingDataException(
    numRows: Int,
    numRequired: Int
) extends Exception(
  s"Bagged training requires at least $numRequired rows, but only $numRows were provided."
) with BaggedTrainingError

case class InsufficientOutputDataException(
    numRows: Int,
    numRequired: Int,
    index: Int
) extends Exception(
  s"Bagged multi-output training requires at least $numRequired rows for each output, " +
    s"but only $numRows were provided for output $index."
) with BaggedTrainingError
