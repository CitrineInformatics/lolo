package io.citrine.lolo.encoders

import org.junit.Test

/**
  * Created by maxhutch on 12/1/16.
  */
@Test
class CategoricalEncoderTest {

  /**
    * Test that encoding preserves equality, unknown goes to zero
    */
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

/** Companion driver */
object CategoricalEncoderTest {
  /**
    * Test driver
    *
    * @param argv args
    */
  def main(argv: Array[String]): Unit = {
    new CategoricalEncoderTest().testStringEncoder()
  }
}
