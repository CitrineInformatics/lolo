package io.citrine.lolo.trees.classification

import io.citrine.lolo.{Learner, TrainingResult, TrainingRow}
import io.citrine.lolo.trees.TrainingLeaf
import io.citrine.random.Random

import scala.collection.mutable

/** Training leaf for a classification tree. */
case class ClassificationTrainingLeaf(
    trainingData: Seq[TrainingRow[Char]],
    trainingResult: TrainingResult[Char],
    depth: Int
) extends TrainingLeaf[Char] {

  override def featureImportance: mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head.inputs.size)(0.0)
}

object ClassificationTrainingLeaf {

  /**
    * @param trainingData for this leaf node
    * @param leafLearner on which to train the data
    * @param depth depth in the tree
    * @param rng random number generator, for reproducibility
    * @return a trained classification leaf node
    */
  def build(
      trainingData: Seq[TrainingRow[Char]],
      leafLearner: Learner[Char],
      depth: Int,
      rng: Random
  ): ClassificationTrainingLeaf =
    ClassificationTrainingLeaf(trainingData, leafLearner.train(trainingData, rng = rng), depth)
}
