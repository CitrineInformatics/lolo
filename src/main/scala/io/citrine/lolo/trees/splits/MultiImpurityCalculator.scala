package io.citrine.lolo.trees.splits

class MultiImpurityCalculator(
                               calculators: Seq[ImpurityCalculator[AnyVal]]
                             ) extends ImpurityCalculator[Array[AnyVal]] {

  def add(value: Array[AnyVal], weight: Double): Double = {
    value.zip(calculators).map{case (v, calc) =>
      if (v.isInstanceOf[Double] && !v.asInstanceOf[Double].isNaN){
        calc.asInstanceOf[ImpurityCalculator[Double]].add(v.asInstanceOf[Double], weight)
      } else if (v.isInstanceOf[Char]) {
        calc.asInstanceOf[ImpurityCalculator[Char]].add(v.asInstanceOf[Char], weight)
      }
    }

    getImpurity
  }

  def remove(value: Array[AnyVal], weight: Double): Double = {
    value.zip(calculators).map{case (v, calc) =>
      if (v.isInstanceOf[Double] && !v.asInstanceOf[Double].isNaN){
        calc.asInstanceOf[ImpurityCalculator[Double]].remove(v.asInstanceOf[Double], weight)
      } else if (v.isInstanceOf[Char]) {
        calc.asInstanceOf[ImpurityCalculator[Char]].remove(v.asInstanceOf[Char], weight)
      }
    }

    getImpurity
  }

  def reset(): Unit = {
    calculators.map(_.reset())
  }

  def getImpurity: Double = {
    calculators.map(_.getImpurity).sum
  }
}

object MultiImpurityCalculator {
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




