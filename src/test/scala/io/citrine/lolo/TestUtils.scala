package io.citrine.lolo

import scala.collection.mutable.ListBuffer

/**
  * Created by maxhutch on 11/28/16.
  */
object TestUtils {

  def readCsv(name: String): Seq[Array[String]] = {
    val res = new ListBuffer[Array[String]]
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    for (line <- bs.getLines()) {
      val cols = line.split(",").map(_.trim)
      res.append(cols)
    }
    bs.close()
    res
  }

  def readCsvAsDoubles(name: String): Seq[Array[Double]] = {
    readCsv(name).map(vec => vec.map(_.toDouble))
  }

}
