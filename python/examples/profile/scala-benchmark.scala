import io.citrine.lolo.learners.RandomForest
import io.citrine.theta.Stopwatch
import scala.collection.mutable.ListBuffer
import java.io.FileInputStream

/**
* Time training and application of models
* @param trainingData which is used both to train and then later apply the models to
* @return the training and application time, in seconds
*/
def timedTest(trainingData: Seq[(Vector[Any], Any)], evalData: Seq[(Vector[Any], Any)]): (Double, Double, Double) = {
    val inputs = runData.map(_._1).toVector
    val baggedLearner = RandomForest.apply(numTrees = trainingData.length)

    val timeTraining = Stopwatch.time({baggedLearner.train(data).getModel()}, benchmark = "None", minRun = 16, targetError = 0.1, maxRun = 32)
    val model = baggedLearner.train(trainingData).getModel()

    val timePredicting = Stopwatch.time({model.transform(inputs).getExpected()}, benchmark = "None", minRun = 16, targetError = 0.1, maxRun = 32)
    val timeUncertainty = Stopwatch.time({model.transform(inputs).getUncertainty()}, benchmark = "None", minRun = 16, targetError = 0.1, maxRun = 32)

    (timeTraining, timePredicting, timeUncertainty)
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
val (train, expect, uncert) = timedTest(data, runData)
println(s"${train},${expect},${uncert}")

