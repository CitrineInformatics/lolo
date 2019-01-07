package io.citrine.lolo

import io.citrine.lolo.stats.functions.Friedman

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
  * Created by maxhutch on 11/28/16.
  */
object TestUtils {

  def readCsv(name: String): Seq[Vector[Any]] = {
    val res = new ListBuffer[Vector[Any]]
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    for (line <- bs.getLines()) {
      val cols = line.split(",").map(_.trim).map { token =>
        try {
          token.toDouble
        } catch {
          case _: Throwable => token
        }
      }.toVector
      res.append(cols)
    }
    bs.close()
    res
  }

  def generateTrainingData(
                            rows: Int,
                            cols: Int,
                            function: (Seq[Double] => Double) = Friedman.friedmanGrosseSilverman,
                            xscale: Double = 1.0,
                            xoff: Double = 0.0,
                            noise: Double = 0.0,
                            seed: Long = 0L
                          ): Vector[(Vector[Double], Double)] = {
    val rnd = new Random(seed)
    Vector.fill(rows) {
      val input = Vector.fill(cols)(xscale * rnd.nextDouble() + xoff)
      (input, function(input) + noise * rnd.nextGaussian())
    }
  }

  def iterateTrainingData(
                           cols: Int,
                           function: (Seq[Double] => Double) = Friedman.friedmanGrosseSilverman,
                           xscale: Double = 1.0,
                           xoff: Double = 0.0,
                           noise: Double = 0.0,
                           seed: Long = 0L
                          ): Iterator[(Vector[Double], Double)] = {
    val rnd = new Random(seed)
    Iterator.continually {
      val input = Vector.fill(cols)(xscale * rnd.nextDouble() + xoff)
      (input, function(input) + noise * rnd.nextGaussian())
    }
  }

  def binTrainingData(continuousData: Seq[(Vector[Double], Double)],
                      inputBins: Seq[(Int, Int)] = Seq(),
                      responseBins: Option[Int] = None
                     ): Seq[(Vector[Any], Any)] = {
    var outputData: Seq[(Vector[Any], Any)] = continuousData
    inputBins.foreach { case (index, nBins) =>
      outputData = outputData.map { case (input, response) =>
        (input.updated(index, Math.round(input(index).asInstanceOf[Double] * nBins).toString), response)
      }
    }
    responseBins.foreach { nBins =>
      val max = continuousData.map(_._2).max
      val min = continuousData.map(_._2).min
      outputData = outputData.map { case (input, response) =>
        (input, Math.round(response.asInstanceOf[Double] * nBins / (max - min)).toString)
      }
    }
    outputData
  }

}
