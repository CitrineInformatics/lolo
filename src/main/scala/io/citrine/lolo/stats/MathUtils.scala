package io.citrine.lolo.stats

object MathUtils {

  /**
    * Compare two Doubles for approximately equality subject to a given tolerance.
    *
    * @param a double to compare
    * @param b double to compare
    * @param tolerance of comparison
    * @return whether a and b are (approximately) equal
    */
  def tolerantEquals(a: Double, b: Double, tolerance: Double = 1e-12): Boolean = {
    if (a.isNaN || b.isNaN) {
      false
    } else if (a.isInfinite || b.isInfinite) {
      a == b
    } else {
      math.abs(a - b) <= tolerance
    }
  }
}
