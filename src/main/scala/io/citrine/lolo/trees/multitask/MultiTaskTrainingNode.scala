package io.citrine.lolo.trees.multitask

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.linear.GuessTheMeanLearner
import io.citrine.lolo.trees.{InternalModelNode, ModelNode, TrainingLeaf}
import io.citrine.lolo.trees.splits.{MultiTaskSplitter, NoSplit}

class MultiTaskTrainingNode(inputs: Seq[(Vector[AnyVal], Array[AnyVal], Double)]) {

  val (split, _) = MultiTaskSplitter.getBestSplit(inputs, inputs.head._1.size, 1)
  val (leftChild: Option[MultiTaskTrainingNode], rightChild: Option[MultiTaskTrainingNode]) = split match {
    case _: NoSplit => (None, None)
    case _: Any =>
      val (leftData, rightData) = inputs.partition(row => split.turnLeft(row._1))
      (Some(new MultiTaskTrainingNode(leftData)), Some(new MultiTaskTrainingNode(rightData)))
  }

  def getNode(index: Int): ModelNode[PredictionResult[Any]] = {
    val label = inputs.head._2(index)
    val reducedData = if (label.isInstanceOf[Double]) {
      inputs.map(x => (x._1, x._2(index).asInstanceOf[Double], x._3)).filterNot(_._2.isNaN)
    } else {
      inputs.map(x => (x._1, x._2(index).asInstanceOf[Char], x._3)).filter(_._2 > 0)
    }
    val (left, right) = reducedData.partition(r => split.turnLeft(r._1))

    val node = if (leftChild.isDefined && rightChild.isDefined && left.nonEmpty && right.nonEmpty) {
      if (label.isInstanceOf[Double]) {
        new InternalModelNode[PredictionResult[Double]](
          split,
          leftChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]],
          rightChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Double]]]
        )
      } else {
        if (!label.isInstanceOf[Char]) throw new IllegalArgumentException("Training data wasn't double or char")
         new InternalModelNode[PredictionResult[Char]](
          split,
          leftChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]],
          rightChild.get.getNode(index).asInstanceOf[ModelNode[PredictionResult[Char]]]
        )
      }
    } else {
      if (label.isInstanceOf[Double]) {
        new TrainingLeaf[Double](reducedData.asInstanceOf[Seq[(Vector[AnyVal], Double, Double)]], new GuessTheMeanLearner(), 1).getNode()
      } else {
        new TrainingLeaf[Char](reducedData.asInstanceOf[Seq[(Vector[AnyVal], Char, Double)]], new GuessTheMeanLearner(), 1).getNode()
      }
    }
    node.asInstanceOf[ModelNode[PredictionResult[Any]]]
  }
}

object MultiTaskTrainingNode