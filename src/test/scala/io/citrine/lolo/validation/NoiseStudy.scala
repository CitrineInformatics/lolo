package io.citrine.lolo.validation

import io.citrine.lolo.bags.{BaggedResult, Bagger}
import io.citrine.lolo.stats.functions.{Friedman, Linear}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.RegressionSplitter
import io.citrine.lolo.{Learner, PredictionResult, TestUtils}
import org.knowm.xchart.BitmapEncoder.BitmapFormat
import org.knowm.xchart.{BitmapEncoder, CategoryChart, CategoryChartBuilder}

import scala.util.Random

object NoiseStudy {
  val rng = new Random(372845L)

  val nTest: Int = 256
  val nRound: Int = 64
  val base: Double = 0.01
  val nFeature: Int = 8

  def main(args: Array[String]): Unit = {

    // testPoint()
    // generateNoiseScan(nTrain = 64, frequency = 0.5)
    // generateNoiseScan(nTrain = 128, frequency = 0.5)
    // generateNoiseScan(nTrain = 256, frequency = 0.5)
    // generateNoiseScan(nTrain = 64, frequency = 1.0)
    // generateNoiseScan(nTrain = 128, frequency = 1.0)
    // generateNoiseScan(nTrain = 256, frequency = 1.0)
    // scanLeafSize()
    // generateNoiseFrequnecyScan(amplitude = 512.0)
    // generateNoiseFrequnecyScan(amplitude = 32.0)
    // generateNoisePhaseScan(amplitude = 16.0, regularization = 0.8)
    // generateNoisePhaseScan(amplitude = 128.0, regularization = 0.8)
    // generateNoisePhaseScan(amplitude = 256.0, regularization = 0.8)
    // generateNoisePhaseScan(amplitude = 512.0, regularization = 0.8)
    /* Today */
    // compareNoise(128, 1.0)
    // compareNoise(128, 0.5)
    // comparePhase(128, 128.0)
    // comparePhase(128, 512.0)
    // comparePhase(128, 8.0)
    // adding(amplitude = 128.0)
    // adding(amplitude = 256.0)
    // adding(amplitude = 512.0)
    // scanLeafSize(0.5, amplitude = 128.0)
    // scanLeafWeight(0.5, amplitude = 128.0)
    // scanLeafWeight(0.75, amplitude = 128.0)
    // scanLeafWeight(0.9, amplitude = 128.0)
    // scanLeafWeight(1.0, amplitude = 128.0)
    // scanRegParam(frequency = 0.5, amplitude = 128.0)
    // scanRegParam(frequency = 0.5, amplitude = 512.0)
    // debug(amplitude = 512.0)
    // compareNoise(1024, 1.0)
    // compareNoiseConfidence(128, 1.0)
    // compareNoiseConfidence(128, 0.5)
    // comparePhaseConfidence(128, 128.0)
    // comparePhaseConfidence(128, 512.0)
    // comparePhaseConfidence(128, 8.0)
    learning_curve(512.0)
  }

  val dataCache = scala.collection.mutable.HashMap.empty[(Int, Double, Double, Double), Vector[(Vector[Double], (Double, Double))]]
  def getData(nFeature: Int, base: Double, amplitude: Double, frequency: Double): Vector[(Vector[Double], (Double, Double))] = {
    if (dataCache.contains((nFeature, base, amplitude, frequency))) {
      dataCache((nFeature, base, amplitude, frequency))
    } else {
      val entry = TestUtils.iterateTrainingData(nFeature, Friedman.friedmanSilverman, seed = rng.nextLong())
        .take(32768)
        .map {
          case (f, l) if rng.nextDouble() < frequency =>
            (f, (l, base * amplitude))
          case (f, l) =>
            (f, (l, base))
        }.toVector
      dataCache((nFeature, base, amplitude, frequency)) = entry
      entry
    }
  }

  def generateNoiseScan(nTrain: Int = 128, frequency: Double = 0.5): Unit = {
    val nTree = 4 * nTrain
    Seq("WEIGHT", "NONE", "DRAW").foreach { noiseMode =>
      val chart = Merit.plotMeritScan(
        s"Amplitude (base = ${base})",
        Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0),
        // Seq(1.0, 32.0, 512.0),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence(false), "RMSE" -> RootMeanSquareError),
        logScale = true,
        yMin = Some(0.0),
        yMax = Some(1.0),
        rng = rng,
        title = Some(s"Increasing Noise with Frequency = ${frequency}")
      ) { amplitude: Double =>
        val data = getData(nFeature, base, amplitude, frequency)
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature,
            minLeafInstances = 1
          ),
          numBags = nTree.toInt,
          useJackknife = true,
          uncertaintyCalibration = false,
          noiseMode = noiseMode
        )
        StatisticalValidation(rng = rng).generativeValidationWithNoise(
          data,
          learner,
          nTrain = nTrain.toInt,
          nTest = nTest,
          nRound = nRound)
      }
      BitmapEncoder.saveBitmap(chart, s"./scan-amplitude-ntrain-${nTrain}-ntree-${nTree}-frequency-${frequency}-noise-${noiseMode}", BitmapFormat.PNG)
    }
  }

  def generateNoiseFrequnecyScan(amplitude: Double = 32.0): Unit = {
    val nTrain = 128
    // Seq("NONE", "WEIGHT", "DRAW").foreach { noiseMode =>
    Seq("WEIGHT").foreach { noiseMode =>
      val chart = Merit.plotMeritScan(
        "Frequency",
        Seq(0.0, 0.5, 0.75, 0.875, 0.9375),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence(false)),
        logScale = false,
        yMin = Some(0.0),
        yMax = Some(1.0),
        rng = rng
      ) { frequency: Double =>
        println(s"Starting frequency ${frequency}")
        val nTrainAdj = (nTrain / (1.0 - frequency)).toInt
        val nTree = 2 * nTrainAdj
        val data = getData(nFeature, base, amplitude, frequency)
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature,
            minLeafInstances = 1
          ),
          numBags = nTree,
          useJackknife = true,
          uncertaintyCalibration = false,
          noiseMode = noiseMode
        )
        StatisticalValidation(rng = rng).generativeValidationWithNoise(
          data,
          learner,
          nTrain = nTrainAdj,
          nTest = nTest,
          nRound = nRound)
      }
      BitmapEncoder.saveBitmap(chart, s"./scan-frequency-amp-${amplitude}-ntrain-${nTrain}-noise-${noiseMode}", BitmapFormat.PNG)
    }
  }

  def generateNoisePhaseScan(amplitude: Double = 32.0, regularization: Double = 0.0): Unit = {
    val nTrain = 128
    val nTree = 4 * nTrain
    // Seq("NONE", "WEIGHT", "DRAW", "HYBRID").foreach { noiseMode =>
    Seq("DRAW", "HYBRID").foreach { noiseMode =>
      val chart = Merit.plotMeritScan(
        "% Noisy",
        // Seq(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        Seq(0.0, 1.0),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence(false), "RMSE" -> RootMeanSquareError),
        logScale = false,
        yMin = Some(0.0),
        yMax = Some(1.0),
        rng = rng
      ) { frequency: Double =>
        val data = getData(nFeature, base, amplitude, frequency)
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature,
            minLeafInstances = 1
          ),
          numBags = nTree,
          useJackknife = true,
          uncertaintyCalibration = false,
          noiseMode = noiseMode,
          weightRegularization = Some(regularization)
        )
        StatisticalValidation(rng = rng).generativeValidationWithNoise(
          data,
          learner,
          nTrain = nTrain,
          nTest = nTest,
          nRound = nRound)
      }
      println(s"Failure rate is: ${BaggedResult.failureCount.toDouble / (BaggedResult.successCount + BaggedResult.failureCount)}")
      BitmapEncoder.saveBitmap(chart, s"./scan-phase4-amp-${amplitude}-ntrain-${nTrain}-nTree-${nTree}-noise-${noiseMode}-reg-${regularization}", BitmapFormat.PNG)
    }
  }

  def scanRegParam(amplitude: Double = 128.0, frequency: Double = 0.5): Unit = {
    val nTrain = 128
    val nTree = 4 * nTrain
    val options = Seq("WEIGHT", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
    val chart = Merit.plotMeritScanComparison(
      s"Reg Param",
      Seq(0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0),
      RootMeanSquareError,
      options,
      logScaleX = true,
      logScaleY = false,
      yMin = Some(0.0),
      rng = rng,
      title = Some(s"Increasing Regularization with Amplitude = ${amplitude * base} and Frequency = ${frequency}")
    ) { regParam: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      val adjustedBaggers = baggers.map{bagger =>
        bagger.copy(
          weightRegularization = Some(regParam)
        )
      }

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        adjustedBaggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./compare-reg-param-amp-${amplitude}-frequency-${frequency}-ntrain-${nTrain}-nTree-${nTree}", BitmapFormat.PNG)
  }

  def scanLeafWeight(frequency: Double = 1.0, amplitude: Double = 512.0): Unit = {
    val nTrain = 128
    val nTree = 4 * nTrain
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
    val chart = Merit.plotMeritScanComparison(
      s"Min Leaf Weight",
      Seq(0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
      // Seq(1.0, 32.0, 512.0),
      RootMeanSquareError,
      options,
      logScaleX = true,
      logScaleY = false,
      yMin = Some(0.0),
      rng = rng,
      title = Some(s"Increasing Minimum Leaf Weight with Amplitude = ${amplitude * base} and Frequency = ${frequency}")
    ) { minWeight: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      val adjustedBaggers = baggers.map{bagger =>
        bagger.copy(
          method = RegressionTreeLearner(numFeatures = nFeature, splitter = RegressionSplitter(minWeight = minWeight))
        )
      }

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        adjustedBaggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./compare-min-weight-amp-${amplitude}-frequency-${frequency}-ntrain-${nTrain}-nTree-${nTree}", BitmapFormat.PNG)
  }

  def scanLeafSize(frequency: Double = 1.0, amplitude: Double = 512.0): Unit = {
    val nTrain = 128
    val nTree = 4 * nTrain
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
    val chart = Merit.plotMeritScanComparison(
      s"Min Leaf Size",
      Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
      RootMeanSquareError,
      options,
      logScaleX = true,
      logScaleY = false,
      yMin = Some(0.0),
      rng = rng,
      title = Some(s"Increasing Minimum Leaf Size with Amplitude = ${amplitude * base} and Frequency = ${frequency}")
    ) { minSize: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      val adjustedBaggers = baggers.map{bagger =>
        bagger.copy(
          method = RegressionTreeLearner(numFeatures = nFeature, minLeafInstances = minSize.toInt)
        )
      }

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        adjustedBaggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./compare-min-size-amp-${amplitude}-frequency-${frequency}-ntrain-${nTrain}-nTree-${nTree}", BitmapFormat.PNG)
  }

  def testPoint(): Unit = {
    val amplitude = 256.0
    val nTrain = 128
    // Seq("NONE", "WEIGHT", "DRAW").foreach { noiseMode =>
    Seq("NONE", "DRAW").foreach { noiseMode =>
      val chart = Merit.plotMeritScan(
        "Frequency",
        // Seq(0.0, 0.75),
        Seq(0.0, 1.0),
        Map("R2" -> CoefficientOfDetermination, "confidence" -> StandardConfidence(false), "RMSE" -> RootMeanSquareError),
        logScale = false,
        yMin = Some(0.0),
        yMax = Some(1.0),
        rng = rng
      ) { frequency: Double =>
        val nTrainAdj = nTrain
        val nTree = 16 * nTrainAdj
        println(s"Using ${nTree} trees for ${nTrainAdj} rows")
        val data = getData(nFeature, base, amplitude, frequency)
        val learner = Bagger(
          RegressionTreeLearner(
            numFeatures = nFeature,
            minLeafInstances = 4
          ),
          numBags = nTree,
          useJackknife = true,
          uncertaintyCalibration = false,
          noiseMode = noiseMode
        )
        StatisticalValidation(rng = rng).generativeValidationWithNoise(
          data,
          learner,
          nTrain = nTrainAdj,
          nTest = nTest,
          nRound = 8)
      }
      BitmapEncoder.saveBitmap(chart, s"./test-point-${noiseMode}", BitmapFormat.PNG)
    }
  }

  def compareNoise(nTrain: Int = 128, frequency: Double = 0.5): Unit = {
    val nTree = 4 * nTrain
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          minLeafInstances = 1,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
      // Seq("WEIGHT").foreach { noiseMode =>
      val amps = Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0)
      val chart = Merit.plotMeritScanComparison(
        s"Amplitude (base = ${base})",
        // Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0),
        Seq(1.0, 8.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0),
        // Seq(1.0, 32.0, 512.0),
        RootMeanSquareError,
        options,
        logScaleX = true,
        logScaleY = true,
        yMin = Some(0.5),
        yMax = Some(8.0),
        rng = rng,
        title = Some(s"Increasing Noise with Frequency = ${frequency}, N = ${nTrain}")
      ) { amplitude: Double =>
        val data = getData(nFeature, base, amplitude, frequency)

        StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
          data,
          baggers,
          nTrain = nTrain.toInt,
          nTest = nTest,
          nRound = nRound)
      }
    chart.addSeries("Noise Level", amps.toArray, amps.map(_ * base).toArray)
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./compare-amplitude-ntrain-${nTrain}-ntree-${nTree}-frequency-${frequency}", BitmapFormat.PNG)
  }

  def comparePhase(nTrain: Int = 128, amplitude: Double = 128.0): Unit = {
    val nTree = 4 * nTrain
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
    val chart = Merit.plotMeritScanComparison(
      s"Noise Frequency",
      Seq(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
      RootMeanSquareError,
      options,
      logScaleX = false,
      logScaleY = false,
      yMin = Some(0.0),
      rng = rng,
      title = Some(s"Increasing Noise Frequency with Amplitude = ${amplitude * base}")
    ) { frequency: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        baggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./compare-amplitude-${amplitude}-ntrain-${nTrain}-ntree-${nTree}", BitmapFormat.PNG)
  }

  def debug(nTrain: Int = 128, amplitude: Double = 128.0): Unit = {
    val nTree = 4 * nTrain
    val options = Seq("NONE", "HYBRID-AUTO", "HYBRID-HAND")
    val baseMethod = Bagger(
      RegressionTreeLearner(
        numFeatures = nFeature,
        minLeafInstances = 1,
        splitter = RegressionSplitter(minWeight = 0.9)
      ),
      numBags = nTree.toInt,
      useJackknife = true,
      uncertaintyCalibration = false,
      noiseMode = "WEIGHT",
      weightRegularization = None
    )
    val baggers = Seq(
      baseMethod.copy(
        noiseMode = "NONE",
        method = RegressionTreeLearner(minLeafInstances = 2)
      ),
      baseMethod.copy(noiseMode = "HYBRID"),
      baseMethod.copy(
        noiseMode = "HYBRID",
        weightRegularization = Some(2.0),
        method = RegressionTreeLearner(splitter = RegressionSplitter(minWeight = 0.9))
      )
    )

    val chart = Merit.plotMeritScanComparison(
      s"Noise Frequency",
      // Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0),
      Seq(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
      // Seq(1.0, 32.0, 512.0),
      RootMeanSquareError,
      options,
      logScaleX = false,
      logScaleY = false,
      yMin = Some(0.0),
      rng = rng,
      title = Some(s"Increasing Noise Frequency with Amplitude = ${amplitude * base}")
    ) { frequency: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        baggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./debug7", BitmapFormat.PNG)
  }


  def adding(nTrain: Int = 128, amplitude: Double = 128.0): Unit = {
    val nTree = 4 * nTrain
    val nRound = 16
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }

    val chart = Merit.plotMeritScanComparison(
      s"Noisy Points",
      Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0),
      // Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0),
      // Seq(1.0, 32.0, 512.0),
      RootMeanSquareError,
      options,
      logScaleX = true,
      logScaleY = false,
      yMin = Some(0.0),
      rng = rng,
      title = Some(s"Increasing Quantity of Noisy Data with Amplitude = ${amplitude * base} with ${nTrain} Accurate Points")
    ) { noiseCount: Double =>
      val baseData = getData(nFeature, base, amplitude, 0.0)
      val noiseData = getData(nFeature, base, amplitude, 1.0)
      val nTrainTotal = nTrain + noiseCount.toInt
      val nTreeTotal = nTrainTotal * 4
      val adjustedBaggers = baggers.map(_.copy(numBags = nTreeTotal))

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        baseData,
        adjustedBaggers,
        nTrain = nTrainTotal,
        nTest = nTest,
        nRound = nRound,
        frequency = Some(noiseCount / (noiseCount + nTrain)),
        source2 = Some(noiseData)
      )
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./add-noise-data-amplitude-${amplitude}-ntrain-${nTrain}", BitmapFormat.PNG)
  }

  def compareNoiseConfidence(nTrain: Int = 128, frequency: Double = 0.5): Unit = {
    val nTree = 4 * nTrain
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          minLeafInstances = 1,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
    // Seq("WEIGHT").foreach { noiseMode =>
    val amps = Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0)
    val chart = Merit.plotMeritScanComparison(
      s"Amplitude (base = ${base})",
      // Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0),
      Seq(1.0, 8.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0),
      // Seq(1.0, 32.0, 512.0),
      StandardConfidence(),
      options,
      logScaleX = true,
      logScaleY = false,
      yMin = Some(0.0),
      yMax = Some(1.0),
      rng = rng,
      title = Some(s"Increasing Noise with Frequency = ${frequency}, N = ${nTrain}")
    ) { amplitude: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        baggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.addSeries("Noise Level", amps.toArray, amps.map(_ * base).toArray)
    chart.setYAxisTitle("Confidence (1 SD)")
    BitmapEncoder.saveBitmap(chart, s"./compare-confidence-amplitude-ntrain-${nTrain}-ntree-${nTree}-frequency-${frequency}", BitmapFormat.PNG)
  }

  def comparePhaseConfidence(nTrain: Int = 128, amplitude: Double = 128.0): Unit = {
    val nTree = 4 * nTrain
    val options = Seq("NONE", "WEIGHT", "DRAW", "HYBRID")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = nTree.toInt,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }
    val chart = Merit.plotMeritScanComparison(
      s"Noise Frequency",
      Seq(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
      StandardConfidence(),
      options,
      logScaleX = false,
      logScaleY = false,
      yMin = Some(0.0),
      yMax = Some(1.0),
      rng = rng,
      title = Some(s"Increasing Noise Frequency with Amplitude = ${amplitude * base}")
    ) { frequency: Double =>
      val data = getData(nFeature, base, amplitude, frequency)

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        data,
        baggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound)
    }
    chart.setYAxisTitle("Confidence (1 SD)")
    BitmapEncoder.saveBitmap(chart, s"./compare-confidence-amplitude-${amplitude}-ntrain-${nTrain}-ntree-${nTree}", BitmapFormat.PNG)
  }

  def learning_curve(amplitude: Double = 1.0): Unit = {
    val nRound = 16
    val options = Seq("NONE")
    val baggers = options.map{noiseMode =>
      Bagger(
        RegressionTreeLearner(
          numFeatures = nFeature,
          splitter = RegressionSplitter(minWeight = 0.9)
        ),
        numBags = 16,
        useJackknife = true,
        uncertaintyCalibration = false,
        noiseMode = noiseMode,
        weightRegularization = None
      )
    }

    val chart = Merit.plotMeritScanComparison(
      s"Training Count",
      Seq(32.0, 64.0, 128.0, 256.0, 512.0),
      // Seq(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0),
      // Seq(1.0, 32.0, 512.0),
      RootMeanSquareError,
      options,
      logScaleX = true,
      logScaleY = true,
      rng = rng,
      title = Some(s"Learning Curve with Amplitude = ${amplitude * base}")
    ) { nTrain: Double =>
      val nTree = 4 * nTrain.toInt
      val baseData = getData(nFeature, base, amplitude, 1.0)
      val adjustedBaggers = baggers.map(_.copy(numBags = nTree))

      StatisticalValidation(rng = rng).generativeValidationWithNoiseMulti(
        baseData,
        adjustedBaggers,
        nTrain = nTrain.toInt,
        nTest = nTest,
        nRound = nRound,
      )
    }
    chart.setYAxisTitle("RMSE")
    BitmapEncoder.saveBitmap(chart, s"./learning-curve-amplitude-${amplitude}", BitmapFormat.PNG)
  }
}