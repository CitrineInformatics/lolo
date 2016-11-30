package io.citrine.lolo

/**
  * Created by maxhutch on 11/29/16.
  */
trait PredictionResult {
  def getExpected(): Seq[Any]
}

trait withUncertainty {
  def getUncertainty(): Seq[Any]
}

trait withScores {
  def getScores(): Seq[Seq[Double]]
}

trait withGradient {
  def getGradient(): Seq[Vector[Double]]
}
