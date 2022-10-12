package io.citrine.lolo

import io.citrine.random.Random
import org.junit.Before

import scala.util.Try

/**
  * Created by maxhutch on 11/28/16.
  */
object TestUtils {

  def readCsv(name: String): Seq[Vector[Any]] = {
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    val res = bs
      .getLines()
      .flatMap { line =>
        Try(
          line
            .split(",")
            .map(_.trim)
            .map { token =>
              try {
                token.toDouble
              } catch {
                case _: Throwable if token == "NaN" => Double.NaN
                case _: Throwable if token.nonEmpty => token
              }
            }
            .toVector
        ).toOption
      }
      .toVector
    bs.close()
    res
  }
}

/**
  * Mix-in class to facilitate predictable random number streams.
  */
trait SeedRandomMixIn {
  // Reset random number generator.
  var rng: Random = Random(2348752L)

  @Before
  def initializeRandom(): Unit = {
    // Seeds must also be reset for each test so that incremental tests are as predictable as running the full case.
    rng = Random(2348752L)
  }
}
