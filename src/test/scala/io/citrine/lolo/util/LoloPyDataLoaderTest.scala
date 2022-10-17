package io.citrine.lolo.util
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.learners.RandomForestRegressor
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
  def testGetRegression(): Unit = {
    val results = new PredictionResult[Double] {
      override def getExpected(): Seq[Double] = {
        1.0 :: 2.0 :: 3.0 :: Nil
      }

      override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
        Some(0.1 :: 0.2 :: 0.3 :: Nil)
      }
    }

    // Get the expected results
    val means = results.getExpected()
    val sigma = results.getUncertainty().get.asInstanceOf[Seq[Double]]

    // Get the results as a byte array, and convert them back
    val reproMeans = LoloPyDataLoader
      .get1DArray(LoloPyDataLoader.getRegressionExpected(results), getDouble = true, bigEndian = false)
      .asInstanceOf[Seq[Double]]
    assert(reproMeans.zip(means).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)

    // Get the uncertainty as a byte array, and convert them back
    val reproSigma = LoloPyDataLoader
      .get1DArray(LoloPyDataLoader.getRegressionUncertainty(results), getDouble = true, bigEndian = false)
      .asInstanceOf[Seq[Double]]
    assert(reproSigma.zip(sigma).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)
  }

  @Test
  def testGetClassification(): Unit = {
    // Get the base results
    val results = new PredictionResult[Int] {
      override def getExpected(): Seq[Int] = {
        0 :: 1 :: 0 :: 1 :: Nil
      }

      override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = {
        Some(Map(0 -> 1.0) :: Map(1 -> 0.5, 0 -> 0.2, 2 -> 0.3) :: Map(0 -> 1.0) :: Map(1 -> 1.0) :: Nil)
      }

      override def getImportanceScores(): Option[Seq[Seq[Double]]] = {
        Some((0.1 :: 0.2 :: Nil) :: (0.4 :: 0.8 :: Nil) :: (0.1 :: 0.5 :: Nil) :: (0.8 :: 0.2 :: Nil) :: Nil)
      }
    }
    val means = results.getExpected()
    val probs: Seq[Double] = results
      .getUncertainty()
      .get
      .asInstanceOf[Seq[Map[Int, Double]]]
      .flatMap(x => 0.until(3).map(i => x.getOrElse(i, 0.0)))
    assert(Math.abs(probs.sum - 4) < 1e-6)

    // Get the expected class via the utility
    val reproExp = LoloPyDataLoader
      .get1DArray(LoloPyDataLoader.getClassifierExpected(results), getDouble = false, bigEndian = false)
      .asInstanceOf[Seq[Int]]
    assert(reproExp.zip(means).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)

    // Get the probs via the utility
    val reproProbs = LoloPyDataLoader
      .get1DArray(LoloPyDataLoader.getClassifierProbabilities(results, 3), getDouble = true, bigEndian = false)
      .asInstanceOf[Seq[Double]]
    assert(reproProbs.length == 4 * 3)
    assert(probs.zip(reproProbs).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)

    // Get the importances via the utility
    val reproImportances = LoloPyDataLoader
      .get1DArray(
        LoloPyDataLoader.getImportanceScores(results),
        getDouble = true,
        bigEndian = false
      )
      .grouped(2)
      .toVector
      .asInstanceOf[Seq[Seq[Double]]]
    assert(reproImportances.length == 4)
    assert(reproImportances.head.zip(0.1 :: 0.2 :: Nil).map(x => Math.abs(x._1 - x._2)).sum < 1e-6)
  }

  @Test
  def testSerialization(): Unit = {
    val model = new RandomForestRegressor(numTrees = 256)
    val bytes = LoloPyDataLoader.serializeObject(model, 9)
    val model2 = LoloPyDataLoader.deserializeObject(bytes).asInstanceOf[RandomForestRegressor]
    assert(model.numTrees == model2.numTrees)
  }
}
