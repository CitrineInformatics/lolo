package io.citrine.lolo.encoders

import org.junit.{Assert, Test}

import java.io._

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
    assertEncode(encoder, inputs ++ Seq("barfoo"))
    Assert.assertEquals(0, encoder.encode("barfoo"))
  }

  @Test
  def testSerializable(): Unit = {
    val inputs = Seq("foo", "bar", "foobar")
    val encoder: CategoricalEncoder[String] = CategoricalEncoder.buildEncoder(inputs)
    for (k <- encoder.decoding.keySet) {
      encoder.decode(k)
    }
    val deserialized = deserialize(serialize(encoder))
    assertEncode(deserialized, inputs)
  }

  private def assertEncode(encoder: CategoricalEncoder[String], testValues: Seq[String]): Unit = {
    for (test <- testValues) {
      Assert.assertEquals(encoder.encode(test), encoder.encode(test))
    }
  }

  private def serialize(encoder: CategoricalEncoder[String]): Array[Byte] = {
    val outputStream = new ByteArrayOutputStream(512)
    val out = new ObjectOutputStream(outputStream)
    try {
      out.writeObject(encoder)
      outputStream.toByteArray
    } catch {
      case _: IOException => throw new AssertionError("cannot serialize")
    } finally {
      if (out != null) out.close()
    }
  }

  private def deserialize(bytes: Array[Byte]): CategoricalEncoder[String] = {
    val in = new ObjectInputStream(new ByteArrayInputStream(bytes))
    try {
      in.readObject().asInstanceOf[CategoricalEncoder[String]]
    } catch {
      case _: IOException => throw new AssertionError("cannot deserialize")
    } finally {
      if (in != null) in.close()
    }
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
    new CategoricalEncoderTest().testSerializable()
  }
}
