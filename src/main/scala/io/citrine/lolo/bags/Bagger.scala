package io.citrine.lolo.bags

import breeze.linalg.{DenseMatrix}
import breeze.stats.distributions.Poisson
import io.citrine.lolo.{Learner, Model}

import scala.collection.parallel.immutable.ParSeq

/**
  * Created by maxhutch on 11/14/16.
  */
class Bagger(method: Learner, var numBags: Int = -1) extends Learner {
  override def train(trainingData: Seq[(Vector[Any], Any)], weights: Option[Seq[Double]] = None): BaggedModel = {
    val actualBags = if (numBags > 0) {
      numBags
    } else {
      trainingData.size
    }

    val dist = new Poisson(1.0)
    val Nij = DenseMatrix.tabulate(trainingData.size, actualBags) { case (i,j) =>
      dist.draw()
    }
    val models = (0 until  actualBags).par.map{i =>
      method.train(trainingData, Some(Nij(::, i).toArray.map(_.toDouble).toSeq))
    }

    new BaggedModel(models, Nij)
  }
}

class BaggedModel(models: ParSeq[Model], Nij: DenseMatrix[Int]) extends Model {

  override def transform(inputs: Seq[Vector[Any]]): Seq[Any] = {
    val predictions = models.map(model => model.transform(inputs))
    predictions.head.head match {
      case x: Double => predictions.seq.transpose.map(ps => ps.asInstanceOf[Seq[Double]].sum / ps.size)
      case x: Any => predictions.transpose.map(ps => ps.groupBy(identity).mapValues(_.size).maxBy(_._2)._1).seq
    }
  }

  override def getFeatureImportance(): Array[Double] = {
    val importances: Array[Double] = models.map(model => model.getFeatureImportance()).reduce{ (v1, v2) =>
      v1.zip(v2).map(p => p._1 + p._2)
    }
    importances.map(_ / models.size)
  }
}