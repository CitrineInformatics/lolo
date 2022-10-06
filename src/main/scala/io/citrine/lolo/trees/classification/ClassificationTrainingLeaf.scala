package io.citrine.lolo.trees.classification

import io.citrine.lolo.{Learner, TrainingResult}
import io.citrine.lolo.trees.TrainingLeaf
import io.citrine.random.Random

import scala.collection.mutable

/** Training leaf for a classification tree. */
case class ClassificationTrainingLeaf(
    trainingData: Seq[(Vector[AnyVal], Char, Double)],
    trainingResult: TrainingResult,
    depth: Int
) extends TrainingLeaf[Char] {

  override def featureImportance: mutable.ArraySeq[Double] = mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
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
      trainingData: Seq[(Vector[AnyVal], Char, Double)],
      leafLearner: Learner,
      depth: Int,
      rng: Random
  ): ClassificationTrainingLeaf =
    ClassificationTrainingLeaf(trainingData, leafLearner.train(trainingData, rng = rng), depth)
}
