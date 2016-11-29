package io.citrine.lolo.encoders

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by maxhutch on 11/28/16.
  */
class CategoricalEncoder[T](encoding: Map[T, Char]) {

  lazy val decoding: Map[Char, T] = encoding.groupBy(_._2).mapValues(_.keys.head)

  def encode(input: T): Char = encoding.getOrElse(input, 0)

  def decode(output: Char): T = decoding(output)
}

object CategoricalEncoder {
  def buildEncoder[T](values: Seq[T]): CategoricalEncoder[T] = {
    var index: Char = 1 // leave 0 for unknown
    val pairs = ListBuffer.empty[(T, Char)]
    val seen = new mutable.HashSet[T]()
    for (v <- values) {
      if (!seen.contains(v)) {
        seen.add(v)
        pairs.append((v, index))
        index = (index + 1).toChar
      }
    }
    new CategoricalEncoder[T](pairs.toMap)
  }
}
