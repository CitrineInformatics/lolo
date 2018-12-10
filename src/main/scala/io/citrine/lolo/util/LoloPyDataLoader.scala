package io.citrine.lolo.util

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
  * Tool used to transfer data from LoloPy to the JVM
  *
  * Works by receiving an byte array and converting it to Scala Sequence
  */
object LoloPyDataLoader {

  /**
    * Receives the feature array. Assumes it to be in a C-ordering
    * @param input Input array as a byte array
    * @param numAttributes Number of features in the array
    * @param bigEndian Whether the numbers are is big-endian or not
    * @return The array as a Scala array
    */
  def getFeatureArray(input: Array[Byte], numAttributes : Integer, bigEndian: Boolean) : Seq[Vector[Double]] = {
    // Get ordering
    val ordering = if (bigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN

    // Group the array into digits
    val digits = input.grouped(8)

    // Generate digits
    digits.map(ByteBuffer.wrap(_).order(ordering).getDouble).grouped(numAttributes).map(_.toVector).toSeq
  }


  /**
    * Receive a class or weight array
    * @param input Input byte array
    * @param getFloat Whether to read doubles (or integers)
    * @param bigEndian Whether the file
    */
  def get1DArray(input: Array[Byte], getFloat: Boolean, bigEndian: Boolean) : Seq[Any] = {
    // Get ordering
    val ordering = if (bigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN

    // Convert integer or
    if (getFloat) {
      input.grouped(8).map(ByteBuffer.wrap(_).order(ordering).getDouble).toSeq
    } else {
      input.grouped(4).map(ByteBuffer.wrap(_).order(ordering).getInt).toSeq
    }
  }

  def zipTrainingData(X: Seq[Vector[Double]], y: Seq[Any]) : Seq[(Vector[Double], Any)] = {
    X.zip(y)
  }
}
