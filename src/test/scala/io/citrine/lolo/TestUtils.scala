package io.citrine.lolo

import io.citrine.lolo.stats.functions.Friedman

import scala.annotation.tailrec
import scala.util.{Random, Try}

/**
  * Created by maxhutch on 11/28/16.
  */
object TestUtils {

  def readCsv(name: String): Seq[Vector[Any]] = {
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    val res = bs.getLines().flatMap{line =>
      Try(line.split(",").map(_.trim).map { token =>
        try {
          token.toDouble
        } catch {
          case _: Throwable if token == "NaN" => Double.NaN
          case _: Throwable if token.nonEmpty => token
        }
      }.toVector).toOption
    }.toVector
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
                            heteroscedastic: Boolean = false,
                            seed: Long = 0L,
                            shape: String = "uniform"
                          ): Vector[(Vector[Double], Double)] = {
    val rnd = new Random(seed)
    Vector.fill(rows) {
      val input = if (shape == "uniform") {
        Vector.fill(cols)(xscale * rnd.nextDouble() + xoff)
      } else {
        Vector.fill(cols)(xscale * rnd.nextGaussian() / 0.5 + xoff)
      }
      if (heteroscedastic) {
        val r2 = 2.0 * Math.sqrt(input.map(x => x*x).sum / input.size)
        (input, function(input) + noise * rnd.nextGaussian() * r2)
      } else {
        (input, function(input) + noise * rnd.nextGaussian())
      }
    }
  }

  @tailrec
  private def makeGrid(dims: Seq[Int], partial: Vector[Vector[Double]] = Vector.empty): Vector[Vector[Double]] = {
    if (dims.isEmpty) {
      return partial
    }

    val num = dims.last
    val nextDim = if (num == 1) {
      Vector(0.5)
    } else {
      Vector.tabulate(num){i => i.toDouble / (num - 1)}
    }

    val updated = if (partial.isEmpty) {
      nextDim.map(x => Vector(x))
    } else {
      nextDim.flatMap { x =>
        partial.map(p => Vector(x) ++ p)
      }
    }

    makeGrid(dims.init, updated)
  }

  def generateGridData(
                        dims: Seq[Int],
                        function: (Seq[Double] => Double) = Friedman.friedmanGrosseSilverman,
                        xscale: Double = 1.0,
                        xoff: Double = 0.0,
                        noise: Double = 0.0,
                        seed: Long = 0L
                      ): Vector[(Vector[Double], Double)] = {
    val rnd = new Random(seed)

    makeGrid(dims).map{x =>
      val input = x.map(_ * xscale + xoff)
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
