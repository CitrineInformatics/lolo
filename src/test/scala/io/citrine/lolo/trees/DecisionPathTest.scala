package io.citrine.lolo.trees

import org.junit.Test
import org.scalatest.Assertions._

@Test
class DecisionPathTest {
  @Test
  def testExtend() {
    val path = new DecisionPath(2)
    assert(path.size == -1)

    path.extend(1.0, 1.0)
    assert(path.size == 0)
    assert(path.weightBySubsetSize(0) == 1.0)

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

  @Test
  def testCopy(): Unit = {
    val path = new DecisionPath(3)
    var newPath = path.copy()
    assert(path.size == newPath.size)

    path.extend(0.99, 1.0)
    assert(path.size == newPath.size + 1)

    path.extend(0.5, 0.0)
    assert(path.size == newPath.size + 2)

    path.extend(0.1, 1.0)
    assert(path.size == newPath.size + 3)

    newPath = path.copy()
    path.weightBySubsetSize.zip(newPath.weightBySubsetSize).foreach { case (p, np) =>
      assert(p == np)
    }
  }

  @Test
  def testUnwind(): Unit = {
    val path = new DecisionPath(2)
    val path0 = path.copy()
    path.extend(0.9, 1.0)
    val path1 = path.copy()
    path.extend(0.5, 1.0)
    val path2 = path.copy()
    path.extend(0.1, 0.0)
    val path3 = path.copy()

    var unwound = path3.unwind(0.1, 0.0)
    assert(unwound.size == path2.size)

    unwound.weightBySubsetSize.zip(path2.weightBySubsetSize).take(unwound.size + 1).foreach { case (a, b) =>
      assert(a == b)
    }

    unwound = path3.unwind(0.5, 1.0)
    assert(unwound.weightBySubsetSize(0) == 0.05)
    assert(unwound.weightBySubsetSize(1) == 0.0)
  }
}
