package io.citrine.lolo.encoders

import org.junit.Test

/**
  * Created by maxhutch on 12/1/16.
  */
@Test
class CategoricalEncoderTest {

  @Test
  def testStringEncoder(): Unit = {
    val inputs = Seq("foo", "bar", "foobar")
    val encoder = CategoricalEncoder.buildEncoder(inputs)
    val testValues = inputs ++ Seq("barfoo")
    for (test <- testValues) {
      assert(encoder.encode(test) == encoder.encode(test))
    }
    assert(encoder.encode("barfoo") == 0)
  }

}

object CategoricalEncoderTest {
  def main(argv: Array[String]): Unit = {
    new CategoricalEncoderTest().testStringEncoder()
  }
}
