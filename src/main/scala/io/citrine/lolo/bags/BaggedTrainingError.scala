package io.citrine.lolo.bags

sealed abstract class BaggedTrainingError(msg: String) extends Exception(msg) {
  def numRows: Int
  def numRequired: Int
}

case class InsufficientTrainingDataException(
    numRows: Int,
    numRequired: Int
) extends BaggedTrainingError(
      s"Bagged training requires at least $numRequired rows, but only $numRows were provided."
    )

case class InsufficientOutputDataException(
    numRows: Int,
    numRequired: Int,
    index: Int
) extends BaggedTrainingError(
      s"Bagged multi-output training requires at least $numRequired rows for each output, " +
        s"but only $numRows were provided for output $index."
    )
