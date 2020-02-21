package io.citrine.lolo.trees

import org.junit.Test
import org.scalatest.Assertions._

@Test
class FeaturePathTest {
  @Test
  def testExtend() {
    val path = FeaturePath(2)
    assert(path.length == -1)

    path.extend(1.0,1.0,-1)
    assert(path.length == 0)
    assert(path.path(0).featureIndex == -1)
    assert(path.path(0).pathWeight == 1.0)
    assert(path.path(0).zeroFraction == 1.0)
    assert(path.path(0).oneFraction == 1.0)

    path.extend(0.5,1.0,0)
    assert(path.length == 1)
    assert(path.path(0).featureIndex == -1)
    assert(Math.abs(path.path(0).pathWeight - 0.25) < 1e-9)
    assert(Math.abs(path.path(0).zeroFraction - 1.0) < 1e-9)
    assert(Math.abs(path.path(0).oneFraction - 1.0) < 1e-9)
    assert(path.path(1).featureIndex == 0)
    assert(Math.abs(path.path(1).pathWeight - 0.5) < 1e-9)
    assert(Math.abs(path.path(1).zeroFraction - 0.5) < 1e-9)
    assert(Math.abs(path.path(1).oneFraction - 1.0) < 1e-9)

    path.extend(0.0,0.25,1)
    assert(path.length == 2)
    assert(path.path(0).featureIndex == -1)
    assert(Math.abs(path.path(0).pathWeight - 0.0) < 1e-9)
    assert(Math.abs(path.path(0).zeroFraction - 1.0) < 1e-9)
    assert(Math.abs(path.path(0).oneFraction - 1.0) < 1e-9)
    assert(path.path(1).featureIndex == 0)
    assert(Math.abs(path.path(1).pathWeight - 0.020833333333333332) < 1e-9)
    assert(Math.abs(path.path(1).zeroFraction - 0.5) < 1e-9)
    assert(Math.abs(path.path(1).oneFraction - 1.0) < 1e-9)
    assert(path.path(2).featureIndex == 1)
    assert(Math.abs(path.path(2).pathWeight - 0.08333333333333333) < 1e-9)
    assert(Math.abs(path.path(2).zeroFraction - 0.0) < 1e-9)
    assert(Math.abs(path.path(2).oneFraction - 0.25) < 1e-9)
  }

  @Test
  def testCopy(): Unit = {
    val path = FeaturePath(3)
    var newPath = path.copy()
    assert(path.length == newPath.length)

    path.extend(1.0,1.0,-1)
    assert(path.length == newPath.length + 1)

    path.extend(0.5,0.5,0)
    assert(path.length == newPath.length + 2)

    path.extend(0.0,0.25,1)
    assert(path.length == newPath.length + 3)

    newPath = path.copy()
    path.path.zip(newPath.path).foreach{case (p,np) =>
      assert(p.pathWeight == np.pathWeight)
      np.pathWeight += 1.0
      assert(p.pathWeight != np.pathWeight)
    }
  }

  @Test
  def testUnwind(): Unit = {
    val path = FeaturePath(2)
    val path0 = path.copy()
    path.extend(1.0,1.0,-1)
    val path1 = path.copy()
    path.extend(0.5,0.5,0)
    val path2 = path.copy()
    path.extend(0.0,0.25,1)
    val path3 = path.copy()

    var unwound = path3.unwind(2)
    assert(unwound.length == path2.length)

    unwound.path.zip(path2.path).take(unwound.length+1).foreach{case (a,b)=>
      assert(a.oneFraction == b.oneFraction)
      assert(a.zeroFraction == b.oneFraction)
      assert(a.pathWeight == b.pathWeight)
    }

    unwound = path3.unwind(1)
    assert(unwound.path(0).featureIndex == -1)
    assert(unwound.path(1).featureIndex == 1)
    assert(unwound.path(0).zeroFraction == 1.0)
    assert(unwound.path(1).zeroFraction == 0.0)
    assert(unwound.path(0).oneFraction == 1.0)
    assert(unwound.path(1).oneFraction == 0.25)
    assert(unwound.path(0).pathWeight == 0.0)
    assert(unwound.path(1).pathWeight == 0.125)
  }
}
