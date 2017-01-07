package io.citrine.lolo.stats.functions

/**
  * Created by maxhutch on 1/5/17.
  */
object Friedman {

  /**
    * Test function on X \in [0,1]^5 by Friedman and Silverman (1989)
    *
    * From:
    * Friedman, Jerome H., and Bernard W. Silverman. "Flexible parsimonious smoothing and additive modeling."
    * Technometrics 31, no. 1 (1989): 3-21.
    *
    * @param x input vector of length 5 or more
    * @return test function(x)
    */
  def friedmanSilverman(x: Seq[Double]): Double = {
    0.1 * Math.exp(4.0 * x(0)) + 4.0 / (1 + Math.exp(- 20.0 * (x(1) - 0.5))) + 3.0 * x(2) + 2.0 * x(3) + x(4)
  }

  /**
    * Test function on X \in [0,1]^5 by Friedman, Grosse, and Stuetzle (1983)
    *
    * From:
    * Friedman, Jerome H., Eric Grosse, and Werner Stuetzle. "Multidimensional additive spline approximation."
    * SIAM Journal on Scientific and Statistical Computing 4, no. 2 (1983): 291-301.
    * @param x input vector of length 5 or more
    * @return test function(x)
    */
  def friedmanGrosseSilverman(x: Seq[Double]): Double = {
    10.0 * Math.sin(Math.PI * x(0) * x(1)) + 20.0 * Math.pow(x(2) - 0.5, 2) + 10.0 * x(3) + 5.0 * x(4)
  }
}
