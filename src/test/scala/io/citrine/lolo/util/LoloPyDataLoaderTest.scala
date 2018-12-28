package io.citrine.lolo.util
import io.citrine.lolo.TestUtils
import io.citrine.lolo.learners.RandomForest
import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test

class LoloPyDataLoaderTest {

  private def getData: Array[Byte] = {
    val stream = getClass.getClassLoader.getResourceAsStream("4by4double.bytes")
    val byteArray = new Array[Byte](128)
    stream.read(byteArray)
    byteArray
  }

  @Test
  def testFeatureArray(): Unit = {
    val data = LoloPyDataLoader.getFeatureArray(getData, 4, bigEndian = false)
    assert(data.length == 4)
    assert(data.head.length == 4)
    assert(Math.abs(data.head(1) - 1) < 1e-6, "Array read in wrong order")
    assert(Math.abs((4 until 8).zip(data(1)).map(x => Math.abs(x._1 - x._2)).sum) < 1e-6)
  }

  @Test
  def test1DArrayDouble(): Unit = {
    val data = LoloPyDataLoader.get1DArray(getData, getDouble = true, bigEndian = false).asInstanceOf[Seq[Double]]
    assert(data.length == 16)
    assert(Math.abs(data(1) - 1) < 1e-6, "Array read badly")
  }

  @Test
  def test1DArrayInteger(): Unit = {
    val data = LoloPyDataLoader.get1DArray(getData, getDouble = false, bigEndian = false).asInstanceOf[Seq[Int]]
    assert(data.length == 32)
    assert(data.take(3).sum == 0) // First 3 values are all 0
    assert(data(3) == 1072693248) // Fourth is a big one
  }

  @Test
  def testZipping(): Unit = {
    val X = LoloPyDataLoader.getFeatureArray(getData, 4, bigEndian = false)
    val y = LoloPyDataLoader.get1DArray(getData, getDouble = true, bigEndian = false).asInstanceOf[Seq[Double]]
    val trainData = LoloPyDataLoader.zipTrainingData(X, y)
    assert(trainData.length == 4)
    val (x_i, y_i: Double) = trainData.head
    assert(x_i.length == 4)
    assert(Math.abs(x_i(1) - 1) < 1e-6)
    assert(Math.abs(y_i - 0) < 1e-6)
  }

  @Test
  def testGetRegression(): Unit = {
    // Generate an RF model
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(32, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8))
    )

    val RFMeta = RandomForest.apply(numTrees = trainingData.length * 2)
      .train(trainingData)
    val RF = RFMeta.getModel()

    // Generate the prediction results object
    val results = RF.transform(trainingData.map(_._1))

    // Get the expected results
    val means = results.getExpected().asInstanceOf[Seq[Double]]
    val sigma = results.getUncertainty().get.asInstanceOf[Seq[Double]]

    // Get the results as a byte array, and convert them back
    val reproMeans = LoloPyDataLoader.get1DArray(
      LoloPyDataLoader.getRegressionExpected(results), getDouble = true, bigEndian = false).asInstanceOf[Seq[Double]]
    assert(reproMeans.zip(means).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)

    // Get the uncertainty as a byte array, and convert them back
    val reproSigma = LoloPyDataLoader.get1DArray(
      LoloPyDataLoader.getRegressionUncertainty(results), getDouble = true, bigEndian = false).asInstanceOf[Seq[Double]]
    assert(reproSigma.zip(sigma).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)
  }

  @Test
  def testGetClassification(): Unit = {
    // Make a classification model
    val trainingData = TestUtils.binTrainingData(
      TestUtils.generateTrainingData(32, 12, noise = 0.1, function = Friedman.friedmanSilverman),
      inputBins = Seq((0, 8)), responseBins = Some(8)
    )
    val RFMeta = new RandomForest(numTrees = trainingData.length * 2)
      .train(trainingData.map(x => (x._1, Integer.parseInt(x._2.toString))))
    val RF = RFMeta.getModel()
    val nClasses : Int = trainingData.map(x => Integer.parseInt(x._2.toString)).max + 1

    // Get the base results
    val results = RF.transform(trainingData.map(_._1))
    val means = results.getExpected().asInstanceOf[Seq[Int]]
    val probs : Seq[Double] = results.getUncertainty().get.asInstanceOf[Seq[Map[Int, Double]]].flatMap(
      x => 0.until(nClasses).map(i => x.getOrElse(i, 0.0)))
    assert(Math.abs(probs.sum - trainingData.length) < 1e-6)

    // Get the expected class via the utility
    val reproExp = LoloPyDataLoader.get1DArray(
      LoloPyDataLoader.getClassifierExpected(results), getDouble = false, bigEndian = false).asInstanceOf[Seq[Int]]
    assert(reproExp.zip(means).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)

    // Get the probs via the utility
    val reproProbs = LoloPyDataLoader.get1DArray(
      LoloPyDataLoader.getClassifierProbabilities(results, nClasses), getDouble = true, bigEndian = false).asInstanceOf[Seq[Double]]
    assert(reproProbs.length == 32 * nClasses)
    assert(probs.zip(reproProbs).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)
  }
}
