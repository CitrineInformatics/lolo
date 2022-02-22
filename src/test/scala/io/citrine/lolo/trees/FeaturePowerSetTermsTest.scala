package io.citrine.lolo.trees

import breeze.linalg.{norm, DenseVector}
import io.citrine.lolo.SeedRandomMixIn
import org.junit.Test
import org.scalatest.Assertions._

@Test
class FeaturePowerSetTermsTest extends SeedRandomMixIn {

  /**
    * Test a few simple extend cases explicitly
    */
  @Test
  def testExtend(): Unit = {
    val path = new FeaturePowerSetTerms(2)
    assert(path.size == 0)

    path.extend(0.5, 1.0)
    assert(path.size == 1)
    assert(Math.abs(path.weightBySubsetSize(0) - 0.25) < 1e-9)
    assert(Math.abs(path.weightBySubsetSize(1) - 0.5) < 1e-9)

    path.extend(0.1, 0.0)
    assert(path.size == 2)
    assert(Math.abs(path.weightBySubsetSize(0) - 1.0 / 60.0) < 1e-9)
    assert(Math.abs(path.weightBySubsetSize(1) - 1.0 / 60.0) < 1e-9)
    assert(Math.abs(path.weightBySubsetSize(2) - 0.0) < 1e-9)
  }

  /**
    * Test that the order that we extend by doesn't matter
    */
  @Test
  def testExtendOrderInvariant(): Unit = {
    val N = 8
    val factors = Seq.fill(N)((rng.nextDouble(), if (rng.nextBoolean()) 1.0 else 0.0))

    val results = Seq.fill(N) {
      val set = new FeaturePowerSetTerms(N)
      rng.shuffle(factors).foreach(x => set.extend(x._1, x._2))
      // make dense vectors so we can take norms
      new DenseVector(set.weightBySubsetSize)
    }

    assert(results.forall { x => norm(x - results.head) < 1.0e-9 })
  }

  @Test
  def testCopy(): Unit = {
    val path = new FeaturePowerSetTerms(3)
    var newPath = path.copy()
    assert(path.size == newPath.size)

    path.extend(0.99, 1.0)
    assert(path.size == newPath.size + 1)

    path.extend(0.5, 0.0)
    assert(path.size == newPath.size + 2)

    path.extend(0.1, 1.0)
    assert(path.size == newPath.size + 3)

    newPath = path.copy()
    path.weightBySubsetSize.zip(newPath.weightBySubsetSize).foreach {
      case (p, np) =>
        assert(p == np)
    }
  }

  /**
    * Test that unwind will reverse an extend
    */
  @Test
  def testUnwind(): Unit = {
    val N = 8
    val firstFactors = Seq.fill(N)((rng.nextDouble(), if (rng.nextBoolean()) 1.0 else 0.0))
    val secondFactors = Seq.fill(N)((rng.nextDouble(), if (rng.nextBoolean()) 1.0 else 0.0))

    // add both sets of factors to the set (note that we test that extend order doesn't matter already
    val set = new FeaturePowerSetTerms(firstFactors.size + secondFactors.size)
    firstFactors.foreach(x => set.extend(x._1, x._2))
    secondFactors.foreach(x => set.extend(x._1, x._2))

    // Now only add the second set
    val anotherSet = new FeaturePowerSetTerms(secondFactors.size)
    secondFactors.foreach(x => anotherSet.extend(x._1, x._2))

    val thinedSet = firstFactors.foldLeft(set) {
      case (tmp, x) =>
        tmp.unwind(x._1, x._2)
    }

    val anotherWeights = new DenseVector(anotherSet.weightBySubsetSize)
    val weights = new DenseVector(thinedSet.weightBySubsetSize.take(anotherWeights.length))

    assert(norm(weights - anotherWeights) < 1.0e-9)
    assert(thinedSet.weightBySubsetSize.drop(anotherWeights.size).forall(x => x == 0.0))
    assert(thinedSet.size == anotherSet.size)
  }
}
