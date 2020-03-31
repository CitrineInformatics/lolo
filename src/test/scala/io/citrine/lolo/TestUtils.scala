package io.citrine.lolo

import io.citrine.lolo.stats.functions.Friedman
import org.junit.Test
import org.mockito.Mockito
import org.mockito.MockitoSugar

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

class RandomMocker(baseRng: Random) extends MockitoSugar {
  val m = mock[Random]
  val queue = new mutable.Queue[(String, AnyRef, Int, Seq[AnyRef])]

  private def enqueueForVerification(methodName: String, returnValue: AnyRef, wantedNumberOfInvocations: Int, args: AnyRef*): Unit = {
    queue.enqueue((methodName, returnValue, wantedNumberOfInvocations, args))
  }

  def getMock() = m

  def verify(): Unit = {
    val order = Mockito.inOrder(m)
    queue.foreach { case (methodName, _, wantedNumberOfInvocations, args) =>
      val argTypes = args.map(_.getClass)
      val verification = order.verify(m, Mockito.times(wantedNumberOfInvocations))
      val method = verification.getClass.getMethod(methodName, argTypes: _*)
      method.invoke(verification, args: _*)
    }
    order.verifyNoMoreInteractions()
  }

  def shuffle[T, CC[X] <: TraversableOnce[X]](
                  xs: CC[T],
                  toReturn: Option[CC[T]] = None
                 )(implicit bf: CanBuildFrom[CC[T], T, CC[T]]): CC[T] = {
    val out = toReturn match {
      case Some(shuffled) => shuffled
      case None => baseRng.shuffle(xs)
    }
    enqueueForVerification(methodName = "shuffle", returnValue = out.asInstanceOf[AnyRef], wantedNumberOfInvocations = 1, args = xs.asInstanceOf[AnyRef])
    doReturn(out).when(m).shuffle(xs)
  }

  def nextDouble(toReturn: Option[Double] = None): Double = {
    val out = toReturn match {
      case Some(x) => x
      case None => baseRng.nextDouble()
    }
    enqueueForVerification(methodName = "nextDouble", returnValue = out.asInstanceOf[AnyRef], wantedNumberOfInvocations = 1, args = Seq())
    doReturn(out).when(m).nextDouble()
  }
}

class TestRandomMocker extends MockitoSugar {
  @Test
  def testShuffle(): Unit = {
    def mocker = new RandomMocker(Random)
    val itemsToShuffle = (1 to 10).toVector
    val expected = mocker.shuffle(itemsToShuffle)
    val actual = mocker.getMock().shuffle(itemsToShuffle)
    expected.zip(actual).foreach { case (e,a) =>
      assert(e == a, "RandomMocker.shuffle(...) did not return the same value as RandomMocker.getMock().shuffle(...)")
    }
  }
}
