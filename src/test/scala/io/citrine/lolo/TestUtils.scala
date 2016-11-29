package io.citrine.lolo

import scala.collection.mutable.ListBuffer

/**
  * Created by maxhutch on 11/28/16.
  */
object TestUtils {

  def readCsv(name: String): Seq[Vector[Any]] = {
    val res = new ListBuffer[Vector[Any]]
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    for (line <- bs.getLines()) {
      val cols = line.split(",").map(_.trim).map{token =>
        try {
          token.toDouble
        } catch {
          case _ => token
        }
      }.toVector
      res.append(cols)
    }
    bs.close()
    res
  }

}
