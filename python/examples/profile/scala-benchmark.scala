import io.citrine.lolo.learners.RandomForest
import io.citrine.theta.Stopwatch

import scala.collection.mutable.ListBuffer
import java.io.FileInputStream

import io.citrine.lolo.PredictionResult

/**
  * Time training and application of models
  * @param trainingData which is used both to train and then later apply the models to
  * @return the training and application time, in seconds
  */
def timedTest(trainingData: Seq[(Vector[Any], Double)], evalData: Seq[(Vector[Any], Double)]): (Double, Double, Double, Double, Integer) = {
  val inputs = evalData.map(_._1).toVector
  val baggedLearner = new RandomForest(trainingData.length)

  // Do a dry run 
  val model = baggedLearner.train(trainingData).getModel()
  model.transform(inputs).getExpected().toVector
  model.transform(inputs).getUncertainty().toVector

  // Time it for real
  val timeTraining = Stopwatch.time({baggedLearner.train(trainingData).getModel()}, benchmark = "None", minRun = 16, targetError = 0.1, maxRun = 32)
  val timePredicting = Stopwatch.time({model.transform(inputs).getExpected()}, benchmark = "None", minRun = 16, targetError = 0.1, maxRun = 32)
  val timeUncertainty = Stopwatch.time({model.transform(inputs).getUncertainty()}, benchmark = "None", minRun = 16, targetError = 0.1, maxRun = 32)

  // Compute the RMSE and fraction inside 1 std
  val predResult = model.transform(inputs).asInstanceOf[PredictionResult[Double]]
  val error = evalData.map(_._2).zip(predResult.getExpected()).map(x => x._1 - x._2)
  val rmse = Math.sqrt(error.map(x => x * x).sum / error.length)
  val num_below = error.zip(predResult.getUncertainty().get.asInstanceOf[Seq[Double]]).map(x => Math.abs(x._1) < x._2).map(if (_) 1 else 0).sum

  (timeTraining, timePredicting, timeUncertainty, rmse, num_below)
}

def readCsv(name: String): Seq[Vector[Any]] = {
  val res = new ListBuffer[Vector[Any]]
  val stream = new FileInputStream(name)
  val bs = scala.io.Source.fromInputStream(stream)
  for (line <- bs.getLines()) {
    val cols = line.split(",").map(_.trim).map { token =>
      try {
        token.toDouble
      } catch {
        case _: Throwable => token
      }
    }.toVector
    res.append(cols)
  }
  bs.close()
  res.toVector
}

def getTrainingDataFromCsv(name: String): Vector[(Vector[Double], Double)] = {
  val raw = readCsv(name)
  raw.collect {
    case row: Vector[Any] if row.forall(_.isInstanceOf[Double]) =>
      val typed = row.asInstanceOf[Vector[Double]]
      (typed.init, typed.last)
  }.toVector
}

val data = getTrainingDataFromCsv(args(0))
val runData = getTrainingDataFromCsv(args(1))
val (train, expect, uncert, rmse, n_below) = timedTest(data, runData)
println(s"${train},${expect},${uncert},${rmse},${n_below}")
