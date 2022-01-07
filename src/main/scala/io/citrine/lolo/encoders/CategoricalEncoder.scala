package io.citrine.lolo.encoders

/** Encoder that maps a categorical variable to a char Created by maxhutch on 11/28/16.
  *
  * @param encoding
  *   underlying map to use
  * @tparam T
  *   type of the categorical variable
  */
class CategoricalEncoder[T](encoding: Map[T, Char]) extends Serializable {

  /** Inverse of the encoding */
  lazy val decoding: Map[Char, T] = encoding
    .groupBy { case (_, value) => value }
    .map { case (key, value) => key -> value.keys.head }

  /** Just call the encoding. Use 0 for unknown inputs
    *
    * @param input
    *   to encode
    * @return
    *   encoded input as a char
    */
  def encode(input: T): Char = encoding.getOrElse(input, 0)

  /** Just call the decoding.
    *
    * @param output
    *   to decode
    * @return
    *   decoded output
    */
  def decode(output: Char): T = decoding(output)
}

/** Companion object */
object CategoricalEncoder {

  /** Build an encoder from a list of input values
    *
    * @param values
    *   to encode
    * @tparam T
    *   type of the encoder
    * @return
    *   an encoder for those inputs
    */
  def buildEncoder[T](values: Seq[T]): CategoricalEncoder[T] = {
    new CategoricalEncoder[T](values.distinct.zipWithIndex.map(p => (p._1, (p._2 + 1).toChar)).toMap)
  }

  /** Apply a sequence of encoders to transform categorical variables into chars
    *
    * @param input
    *   to encode
    * @param encoders
    *   sequence of encoders
    * @return
    *   input with categoricals encoded as chars
    */
  def encodeInput(input: Vector[Any], encoders: Seq[Option[CategoricalEncoder[Any]]]): Vector[AnyVal] = {
    input.zip(encoders).map { case (v, e) =>
      e match {
        case Some(x) => x.encode(v)
        case None    => v.asInstanceOf[AnyVal]
      }
    }
  }
}
