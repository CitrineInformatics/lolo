package io.citrine.lolo.metrics

import breeze.linalg.{DenseMatrix, sum}

/**
  * Created by maxhutch on 12/28/16.
  */
object ClassificationMetrics {

  def f1scores(predictedVsActual: Seq[(Vector[Any], Any, Any)]): Double = {
      val labels = predictedVsActual.map(_._3).distinct
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
}
