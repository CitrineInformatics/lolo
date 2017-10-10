package io.citrine.lolo.trees.splits

/**
  * Compute the impurity of array-valued labels by summing over the impurity of each index on its own
  * @param calculators to use to compute the impurity of each index
  */
class MultiImpurityCalculator(
                               calculators: Seq[ImpurityCalculator[AnyVal]]
                             ) extends ImpurityCalculator[Array[AnyVal]] {

  /**
    * Add the value to each calculator
    * @param value to add
    * @param weight of the value
    * @return the impurity after adding
    */
  def add(value: Array[AnyVal], weight: Double): Double = {
    value.zip(calculators).map{case (v, calc) =>
      if (v.isInstanceOf[Double] && !v.asInstanceOf[Double].isNaN){
        calc.asInstanceOf[ImpurityCalculator[Double]].add(v.asInstanceOf[Double], weight)
      } else if (v.isInstanceOf[Char] && v.asInstanceOf[Char] > 0) {
        calc.asInstanceOf[ImpurityCalculator[Char]].add(v.asInstanceOf[Char], weight)
      }
    }

    getImpurity
  }

  /**
    * Remove the value from each calculator
    * @param value to remove
    * @param weight of the value
    * @return the impurity after removing
    */
  def remove(value: Array[AnyVal], weight: Double): Double = {
    value.zip(calculators).map{case (v, calc) =>
      if (v.isInstanceOf[Double] && !v.asInstanceOf[Double].isNaN){
        calc.asInstanceOf[ImpurityCalculator[Double]].remove(v.asInstanceOf[Double], weight)
      } else if (v.isInstanceOf[Char] && v.asInstanceOf[Char] > 0) {
        calc.asInstanceOf[ImpurityCalculator[Char]].remove(v.asInstanceOf[Char], weight)
      }
    }

    getImpurity
  }

  /**
    * Reset each calculator
    */
  def reset(): Unit = {
    calculators.foreach(_.reset())
  }

  /**
    * Get the impurity as the sum of the impurities
    * @return impurity
    */
  def getImpurity: Double = {
    calculators.map(_.getImpurity).sum
  }
}

/**
  * Companion object
  */
object MultiImpurityCalculator {
  /**
    * Build the calculators for each index and then wrap them in the MultiImpurityCalculator
    * @param labels that have Array values
    * @param weights which are assumed to be constant over the labels at each row
    * @return MultiImpurityCalculator that sum the impurity of each label index
    */
  def build(labels: Seq[Array[AnyVal]], weights: Seq[Double]): MultiImpurityCalculator = {
    val calculators: Seq[ImpurityCalculator[AnyVal]] = labels.transpose.map{labelSeq =>
      if (labelSeq.head.isInstanceOf[Double]) {
        VarianceCalculator.build(labelSeq.asInstanceOf[Seq[Double]], weights)
          .asInstanceOf[ImpurityCalculator[AnyVal]]
      } else {
        GiniCalculator.build(labelSeq.asInstanceOf[Seq[Char]].zip(weights))
          .asInstanceOf[ImpurityCalculator[AnyVal]]
      }
    }
    new MultiImpurityCalculator(calculators)
  }
}




