package io.citrine.lolo

import io.citrine.lolo.stats.functions.Friedman

import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
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

object RandomMethod extends Enumeration {
  val shuffle = Value("Shuffle")
  val nextGaussian = Value("nextGaussian")
  val nextDouble = Value("nextDouble")
}

object RandomLoggerBehavior extends Enumeration {
  val LogCall = Value(1)
  val LogArgs = Value(2)
  val LogResult = Value(4)
}

/**
class MockRandom extends Random {
  private val log = mutable.Queue[(RandomMethod.Value, Option[AnyVal], Option[AnyVal])]
  private val expectations = mutable.Queue[(RandomMethod.Value, Option[AnyVal], Option[AnyVal])]

  def expectShuffle[T, CC[X] <: TraversableOnce[X]](toShuffle: CC[T], out: CC[T])(implicit bf: CanBuildFrom[CC[T], T, CC[T]]): Unit = {
    expectations.apply((RandomMethod.shuffle, Some(toShuffle), Some(out)))
  }

  def expect(method: RandomMethod.Value, args: Option[AnyVal], result: Option[AnyVal]): Unit = {
    expectations.andThen((method, args, result))
  }

  override def shuffle[T, CC[X] <: TraversableOnce[X]](xs: CC[T])(implicit bf: CanBuildFrom[CC[T], T, CC[T]]): CC[T] = {
    val res = super.shuffle(xs)
    logCall(RandomMethod.shuffle, Option[AnyVal], Option[AnyVal])
    res
  }

  override def nextDouble(): Double = {
    super.nextDouble()
  }
}
*/
