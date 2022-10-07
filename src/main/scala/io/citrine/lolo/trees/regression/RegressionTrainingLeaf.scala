package io.citrine.lolo.trees.regression

import io.citrine.lolo.trees.TrainingLeaf
import io.citrine.lolo.{Learner, TrainingResult}
import io.citrine.random.Random

import scala.collection.mutable

/** Training leaf node for regression trees. */
case class RegressionTrainingLeaf(
    trainingData: Seq[(Vector[AnyVal], Double, Double)],
    trainingResult: TrainingResult[Double],
    depth: Int
) extends TrainingLeaf[Double] {

  /**
    * Pull the leaf model's feature importance and rescale it by the remaining impurity
    *
    * @return feature importance as a vector
    */
  override def featureImportance: scala.collection.mutable.ArraySeq[Double] = {
    trainingResult.getFeatureImportance() match {
      case Some(x) =>
        // Compute the weighted sum of the label, the square label, and the weights
        val expectations: (Double, Double, Double) = trainingData
          .map {
            case (_, l, w) =>
              (l * w, l * l * w, w)
          }
          .reduce((u: (Double, Double, Double), v: (Double, Double, Double)) => (u._1 + v._1, u._2 + v._2, u._3 + v._3))
        // Use those sums to compute the variance as E[x^2] - E[x]^2
        val impurity =
          Math.max(expectations._2 / expectations._3 - Math.pow(expectations._1 / expectations._3, 2.0), 0.0)
        mutable.ArraySeq(x: _*).map(_ * impurity)
      case None => mutable.ArraySeq.fill(trainingData.head._1.size)(0.0)
    }
  }
}

object RegressionTrainingLeaf {

  /**
    * @param trainingData for this leaf node
    * @param leafLearner on which to train the data
    * @param depth depth in the tree
    * @param rng random number generator, for reproducibility
    * @return a trained regression leaf node
    */
  def build(
      trainingData: Seq[(Vector[AnyVal], Double, Double)],
      leafLearner: Learner[Double],
      depth: Int,
      rng: Random
  ): RegressionTrainingLeaf =
    RegressionTrainingLeaf(trainingData, leafLearner.train(trainingData, rng = rng), depth)
}
