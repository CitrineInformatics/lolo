package io.citrine.lolo.util

import java.nio.ByteBuffer
import java.nio.ByteOrder

import io.citrine.lolo.PredictionResult

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

  /**
    * Zips the features and labels together
    * @param X Feature array
    * @param y Label array
    * @return Zipped arrays
    */
  def zipTrainingData(X: Seq[Vector[Double]], y: Seq[Any]) : Seq[(Vector[Double], Any)] = {
    X.zip(y)
  }

  /**
    * Generate the results of a regression model, which are assumed to be doubles
    * @param predictionResult Prediction result object
    * @return Byte array of doubles in native system order
    */
  def getRegressionExpected(predictionResult: PredictionResult[Any]) : Array[Byte] = {
    val predResults : Seq[Double] = predictionResult.getExpected().asInstanceOf[Seq[Double]]
    val buffer : ByteBuffer = ByteBuffer.allocate(predResults.length * 8).order(ByteOrder.nativeOrder())
    predResults.foreach(buffer.putDouble)
    buffer.array()
  }

  /**
    * Generate the uncertainties of a regression model, which are assumed to be doubles
    * @param predictionResult Prediction result object
    * @return Byte array of doubles in native system order
    */
  def getRegressionUncertainty(predictionResult: PredictionResult[Any]): Array[Byte] = {
    val predResults : Seq[Double] = predictionResult.getUncertainty().get.asInstanceOf[Seq[Double]]
    val buffer : ByteBuffer = ByteBuffer.allocate(predResults.length * 8).order(ByteOrder.nativeOrder())
    predResults.foreach(buffer.putDouble)
    buffer.array()
  }

  /**
    * Get the predicted class from a classification model
    *
    * Note: The class labels must be integers
    *
    * @param predictionResult Prediction result from a classifier
    * @return Bytes of a integer array of the predicted class labels
    */
  def getClassifierExpected(predictionResult: PredictionResult[Any]): Array[Byte] = {
    val expect = predictionResult.getExpected().asInstanceOf[Seq[Int]]
    val buffer = ByteBuffer.allocate(expect.length * 4).order(ByteOrder.nativeOrder())
    expect.foreach(buffer.putInt)
    buffer.array
  }

  /**
    * Get the classification probabilities as an array
    *
    * Note: Model must be trained with sequential nonnegative integers starting at 0 as class names.
    *
    * @param predictionResult Prediction result from a classifier
    * @param nClasses Number of classes in the problem
    * @return Bytes of a double array of probabilities, stored row-wise (i.e., first entry, then second, ...)
    */
  def getClassifierProbabilities(predictionResult: PredictionResult[Any], nClasses: Int): Array[Byte] = {
    // Get an iterator over the number of classes
    val classes = 0 until nClasses
    val probs = predictionResult.getUncertainty().get.asInstanceOf[Seq[Map[Int, Double]]].map(
      x => classes.map(i => x.getOrElse(i, 0.0))
    )
    val buffer = ByteBuffer.allocate(nClasses * probs.length * 8).order(ByteOrder.nativeOrder())
    probs.flatten.foreach(buffer.putDouble)
    buffer.array
  }
}
