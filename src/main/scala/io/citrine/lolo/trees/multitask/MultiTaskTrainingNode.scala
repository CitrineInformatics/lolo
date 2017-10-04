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

  lazy val leftNodes = leftChild.get.getNodes()
  lazy val rightNodes = rightChild.get.getNodes()


  def getNodes(): Seq[ModelNode[PredictionResult[Any]]] = {
    val labels = inputs.head._2
    labels.indices.map{k =>
      val node = if (leftChild.isDefined && rightChild.isDefined) {
        if (labels(k).isInstanceOf[Double]) {
          new InternalModelNode[PredictionResult[Double]](
            split,
            leftNodes(k).asInstanceOf[ModelNode[PredictionResult[Double]]],
            rightNodes(k).asInstanceOf[ModelNode[PredictionResult[Double]]]
          )
        } else {
          if (!labels(k).isInstanceOf[Char]) throw new IllegalArgumentException("Training data wasn't double or char")
           new InternalModelNode[PredictionResult[Char]](
            split,
            leftNodes(k).asInstanceOf[ModelNode[PredictionResult[Char]]],
            rightNodes(k).asInstanceOf[ModelNode[PredictionResult[Char]]]
          )
        }
      } else {
        if (labels(k).isInstanceOf[Double]) {
          new TrainingLeaf[Double](inputs.map(x => (x._1, x._2(k).asInstanceOf[Double], x._3)), new GuessTheMeanLearner(), 1).getNode()
        } else {
          new TrainingLeaf[Char](inputs.map(x => (x._1, x._2(k).asInstanceOf[Char], x._3)), new GuessTheMeanLearner(), 1).getNode()
        }
      }
      node.asInstanceOf[ModelNode[PredictionResult[Any]]]
    }
  }
}