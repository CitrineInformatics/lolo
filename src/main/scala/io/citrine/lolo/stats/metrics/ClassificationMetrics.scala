package io.citrine.lolo.stats.metrics

import breeze.linalg.{DenseMatrix, sum}

/**
  * Created by maxhutch on 12/28/16.
  */
object ClassificationMetrics {

  /**
    * Compute the weighted average f1 score across the labels
    *
    * @param predictedVsActual to evaluate
    * @return the weighted average f1 score
    */
  def f1scores(predictedVsActual: Seq[(Vector[Any], Any, Any)]): Double = {
    // the test set and training set might have different labels, so we might
    // have predictions that are never actual and vice versa
    val labels = (predictedVsActual.map(_._2) ++ predictedVsActual.map(_._3)).distinct
    val index = labels.zipWithIndex.toMap
    val numLabels = labels.size
    val confusionMatrix = DenseMatrix.zeros[Int](numLabels, numLabels)
    predictedVsActual.foreach(p => confusionMatrix(index(p._2), index(p._3)) += 1)
    val f1scores = labels.indices.map { i =>
      val actualPositive: Double = sum(confusionMatrix(::, i))
      val predictedPositive: Double = sum(confusionMatrix(i, ::))
      val precision = if (predictedPositive > 0) confusionMatrix(i, i) / predictedPositive else 1.0
      val recall = if (actualPositive > 0) confusionMatrix(i, i) / actualPositive else 1.0
      if (precision > 0.0 && recall > 0.0) {
        2.0 * precision * recall / (precision + recall) * actualPositive
      } else {
        0.0
      }
    }
    f1scores.sum / predictedVsActual.size
  }

  def f1scores(predicted: Seq[Any], actual: Seq[Any]): Double = {
    f1scores(predicted.zip(actual).map(x => (Vector(), x._1, x._2)))
  }
}
