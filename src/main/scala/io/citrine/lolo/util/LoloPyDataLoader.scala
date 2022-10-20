package io.citrine.lolo.util

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip._
import io.citrine.lolo.{MultiTaskModelPredictionResult, PredictionResult}

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
  def getFeatureArray(input: Array[Byte], numAttributes: Integer, bigEndian: Boolean): Seq[Vector[Double]] = {
    // Get ordering
    val ordering = if (bigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN

    // Wrap the input as a buffer
    val buffer = ByteBuffer.wrap(input).order(ordering)

    // Generate digits
    val nDigits = input.length / 8
    (0 until nDigits).map(x => buffer.getDouble(x * 8)).grouped(numAttributes).map(_.toVector).toVector
  }

  /**
    * Receive a 1D array of floats or integers
    * @param input Input byte array
    * @param getDouble Whether to read doubles (or integers)
    * @param bigEndian Whether the file
    */
  def get1DArray(input: Array[Byte], getDouble: Boolean, bigEndian: Boolean): Seq[Any] = {
    // Get ordering
    val ordering = if (bigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN

    // Make the buffer
    val buffer = ByteBuffer.wrap(input).order(ordering)

    // Convert to integer or float
    if (getDouble) {
      val nDigits = input.length / 8
      (0 until nDigits).map(x => buffer.getDouble(x * 8)).toVector
    } else {
      val nDigits = input.length / 4
      (0 until nDigits).map(x => buffer.getInt(x * 4)).toVector
    }
  }

  /**
    * Zips the features and labels together
    *
    * This function was created to provide the ability to easily Zip training and testing data when using Lolo
    * from lolopy. Py4J does not support generic methods well, and this function is a workaround for being able
    * to call zip when needed generate a training set.
    *
    * @param X Feature array
    * @param y Label array
    * @return Zipped arrays
    */
  def zipTrainingData(X: Seq[Vector[Double]], y: Seq[Any]): Seq[(Vector[Double], Any)] = {
    X.zip(y)
  }

  /**
    * Generate the results of a regression model, which are assumed to be doubles
    * @param predictionResult Prediction result object
    * @return Byte array of doubles in native system order
    */
  def getRegressionExpected(predictionResult: PredictionResult[Any]): Array[Byte] = {
    val predResults: Seq[Double] = predictionResult.expected.asInstanceOf[Seq[Double]]
    send1DArray(predResults)
  }

  /**
    * Generate the results of a multitask regression model, which are assumed to be all doubles
    * @param predictionResult result of predicting on a multitask model
    * @return Byte array of doubles in native system order (the caller must then reshape the result into a 2d array)
    */
  def getMultiRegressionExpected(predictionResult: MultiTaskModelPredictionResult): Array[Byte] = {
    val predResults = predictionResult.expected.asInstanceOf[Seq[Seq[Double]]].flatten
    send1DArray(predResults)
  }

  /**
    * Send the training entry importance scores to the Python client
    * @param predictionResult Prediction result object
    * @return Byte of array of doubles in native system order
    */
  def getImportanceScores(predictionResult: PredictionResult[Any]): Array[Byte] = {
    send1DArray(predictionResult.importanceScores.get.flatten)
  }

  /**
    * Generate the uncertainties of a regression model, which are assumed to be doubles
    * @param predictionResult Prediction result object
    * @return Byte array of doubles in native system order
    */
  def getRegressionUncertainty(predictionResult: PredictionResult[Any]): Array[Byte] = {
    val predResults: Seq[Double] = predictionResult.uncertainty().get.asInstanceOf[Seq[Double]]
    send1DArray(predResults)
  }

  /**
    * Get the uncertainties of a multitask regression model, which are assumed to be all doubles
    * @param predictionResult result of predicting on a multitask model
    * @return Byte array of doubles in native system order (the caller must then reshape the result into a 2d array)
    */
  def getMultiRegressionUncertainty(predictionResult: MultiTaskModelPredictionResult): Array[Byte] = {
    val uncertaintyResults = predictionResult.uncertainty().get.asInstanceOf[Seq[Seq[Double]]].flatten
    send1DArray(uncertaintyResults)
  }

  /**
    * Get the correlation coefficients between the uncertainties of a multitask regression model.
    * By calling this method for all (i, j) pairs, one can construct a correlation matrix.
    * Combined with getMultiRegressionUncertainty, one can construct the covariance matrix.
    *
    * @param predictionResult result of predicting on a multitask model
    * @param i index of the first output
    * @param j index of the second output
    * @return Byte array of doubles in native system order
    */
  def getRegressionCorrelation(predictionResult: MultiTaskModelPredictionResult, i: Int, j: Int): Array[Byte] = {
    val correlationResults = predictionResult.uncertaintyCorrelation(i, j).get
    send1DArray(correlationResults)
  }

  /**
    * Prepare to send a 1D array of Doubles by converting it to a byte array
    * @param data Data to be sent
    * @return Byte array with all the doubles in Seq ordered in system byte order
    */
  def send1DArray(data: Seq[Double]): Array[Byte] = {
    val buffer: ByteBuffer = ByteBuffer.allocate(data.length * 8).order(ByteOrder.nativeOrder())
    data.foreach(buffer.putDouble)
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
    val expect = predictionResult.expected.asInstanceOf[Seq[Int]]
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
    val probs = predictionResult
      .uncertainty()
      .get
      .asInstanceOf[Seq[Map[Int, Double]]]
      .map(x => classes.map(i => x.getOrElse(i, 0.0)))
    val buffer = ByteBuffer.allocate(nClasses * probs.length * 8).order(ByteOrder.nativeOrder())
    probs.flatten.foreach(buffer.putDouble)
    buffer.array
  }

  /**
    * Convert a JVM object to a byte array
    *
    * Used for saving a model in lolopy inside of a pickle file.
    *
    * @param obj Object to be saved
    * @param compressLevel Compression level used to condense the serialized representation
    * @return Object as a serialized byte array
    */
  def serializeObject(obj: Any, compressLevel: Int): Array[Byte] = {
    // Thanks to: https://stackoverflow.com/questions/39369319/convert-any-type-in-scala-to-arraybyte-and-back
    val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
    val compressedStream: DeflaterOutputStream = new DeflaterOutputStream(stream, new Deflater(compressLevel))
    val oos = new ObjectOutputStream(compressedStream)
    oos.writeObject(obj)
    oos.close()
    compressedStream.close()
    stream.toByteArray
  }

  /**
    * Deserialize a JVM object from a byte array
    *
    * @param bytes Bytes to be unserialized
    * @return The object
    */
  def deserializeObject(bytes: Array[Byte]): Any = {
    val stream = new ObjectInputStream(new InflaterInputStream(new ByteArrayInputStream(bytes)))
    val obj = stream.readObject()
    stream.close()
    obj
  }

  /**
    * Create a PredictionResult object from the mean and uncertainty
    * @param thisExpected    Mean of the predictions of a model
    * @param thisUncertainty Uncertainty of the predictions
    * @return Prediction result object
    */
  def makeRegressionPredictionResult(
      thisExpected: Seq[Double],
      thisUncertainty: Seq[Double]
  ): PredictionResult[Double] = {
    new PredictionResult[Double] {
      override def expected: Seq[Double] = thisExpected
      override def uncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = Some(thisUncertainty)
    }
  }
}
