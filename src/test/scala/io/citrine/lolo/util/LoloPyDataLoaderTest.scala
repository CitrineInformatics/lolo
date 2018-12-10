package io.citrine.lolo.util
import org.junit.Test

class LoloPyDataLoaderTest {

  private def getData: Array[Byte] = {
    val stream = getClass.getClassLoader.getResourceAsStream("4by4double.bytes")
    val byteArray = new Array[Byte](128)
    stream.read(byteArray)
    byteArray
  }

  @Test
  def testFeatureArray() = {
    val data = LoloPyDataLoader.getFeatureArray(getData, 4, false)
    assert(data.length == 4)
    assert(data.head.length == 4)
    assert(Math.abs(data.head(1) - 1) < 1e-6, "Array read in wrong order")
  }

  @Test
  def test1DArrayDouble() = {
    val temp = getData
    val data = LoloPyDataLoader.get1DArray(getData, true, false).asInstanceOf[Seq[Double]]
    assert(data.length == 16)
    assert(Math.abs(data(1) - 1) < 1e-6, "Array read badly")
  }

  @Test
  def test1DArrayInteger() = {
    val data = LoloPyDataLoader.get1DArray(getData, false, false).asInstanceOf[Seq[Int]]
    assert(data.length == 32)
    assert(data.take(3).sum == 0) // First 3 values are all 0
    assert(data(3) == 1072693248) // Fourth is a big one
  }
}
