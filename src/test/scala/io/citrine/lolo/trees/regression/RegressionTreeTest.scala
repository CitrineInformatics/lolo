package io.citrine.lolo.trees.regression

import java.io.{File, FileOutputStream, ObjectOutputStream}
import breeze.linalg.DenseMatrix
import io.citrine.lolo.api.TrainingRow
import io.citrine.lolo.{DataGenerator, SeedRandomMixIn, TestUtils}
import io.citrine.lolo.linear.LinearRegressionLearner
import io.citrine.lolo.stats.functions.Friedman
import io.citrine.lolo.trees.splits.{BoltzmannSplitter, RegressionSplitter}
import org.junit.Test
import org.scalatest.Assertions._

@Test
class RegressionTreeTest extends SeedRandomMixIn {

  /**
    * Trivial models with no splits should have finite feature importance.
    */
  @Test
  def testFeatureImportanceNaN(): Unit = {
    val X = Vector.fill(100) {
      val input = Vector.fill(10)(1.0)
      TrainingRow(input, 2.0)
    }

    val DTLearner = RegressionTreeLearner()
    val DTMeta = DTLearner.train(X)
    assert(DTMeta.featureImportance.get.forall(v => !v.isNaN))
  }

  /**
    * Test a simple tree with only real inputs
    */
  @Test
  def testSimpleTree(): Unit = {
    val csv = TestUtils.readCsv("double_example.csv")
    val trainingData = csv.map(vec => TrainingRow(vec.init, vec.last.asInstanceOf[Double]))
    val DTLearner = RegressionTreeLearner()
    val DT = DTLearner.train(trainingData).model

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(Math.abs(row.label - p) < 1.0e-9)
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 3 && d < 9, s"Depth is $d"))
  }

  /**
    * Test a simple tree with only real inputs
    * Even with randomization, the training data should be memorized
    */
  @Test
  def testrandomizePivotLocation(): Unit = {
    val csv = TestUtils.readCsv("double_example.csv")
    val trainingData = csv.map(vec => TrainingRow(vec.init, vec.last.asInstanceOf[Double], 1.0))
    val DTLearner = RegressionTreeLearner(splitter = RegressionSplitter(randomizePivotLocation = true))
    val DT = DTLearner.train(trainingData).model

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(Math.abs(row.label - p) < 1.0e-9)
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 3 && d < 9, s"Depth is $d"))
  }

  /**
    * Test a larger case and time it as a benchmark guideline
    */
  @Test
  def longerTest(): Unit = {
    val trainingData =
      DataGenerator.generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data
    val DTLearner = RegressionTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.model
    (0 until N).map(i => DTLearner.train(trainingData).model)
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(Math.abs(row.label - p) < 1.0e-9)
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 4 && d < 20, s"Depth is ${d}"))

    /* The first feature should be the most important */
    val importances = DTMeta.featureImportance.get
    assert(importances(1) == importances.max)

    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Train with a categorical variable
    */
  @Test
  def testCategorical(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((0, 8)))
      .data

    val DTLearner = RegressionTreeLearner()
    val N = 100
    val start = System.nanoTime()
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.model
    (0 until N).map(i => DTLearner.train(trainingData).model)
    val duration = (System.nanoTime() - start) / 1.0e9

    println(s"Training large case took ${duration / N} s")

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(Math.abs(row.label - p) < 1.0e-9)
    }
    assert(output.gradient.isEmpty)
    output.depth.foreach(d => assert(d > 4 && d < 21, s"Depth is ${d}"))

    /* The first feature should be the most important */
    val importances = DTMeta.featureImportance.get
    assert(importances(1) == importances.max)
    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Train with linear leaves.  This case is under-constriained, so we should hit the results exactly
    */
  @Test
  def testLinearLeaves(): Unit = {
    val trainingData =
      DataGenerator.generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng).data

    val linearLearner = LinearRegressionLearner(regParam = Some(0.0))
    val DTLearner = RegressionTreeLearner(leafLearner = Some(linearLearner), minLeafInstances = 2)
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.model

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) =>
        assert(Math.abs(row.label - p) < 1.0e-9)
    }
    assert(output.gradient.isDefined)
    output.depth.foreach(d => assert(d > 4 && d < 18, s"Depth is $d"))

    /* The first feature should be the most important */
    val importances = DTMeta.featureImportance.get
    assert(importances(1) == importances.max)
    /* They should all be non-zero */
    assert(importances.min > 0.0)

    val tmpFile: File = File.createTempFile("tmp", ".csv")
    val oos = new ObjectOutputStream(new FileOutputStream(tmpFile))
    oos.writeObject(DT)
  }

  /**
    * Test a really short tree to make sure the linear model feature importance gets carried through
    */
  @Test
  def testStumpWithLinearLeaf(): Unit = {
    val trainingData = DataGenerator
      .generate(1024, 12, noise = 0.1, function = Friedman.friedmanSilverman, rng = rng)
      .withBinnedInputs(bins = Seq((11, 8)))
      .data

    val linearLearner = LinearRegressionLearner(regParam = Some(1.0))
    val DTLearner = RegressionTreeLearner(leafLearner = Some(linearLearner), maxDepth = 0)
    val DTMeta = DTLearner.train(trainingData)
    val DT = DTMeta.model

    val linearImportance = linearLearner.train(trainingData).featureImportance.get

    val importances = DTMeta.featureImportance.get

    /* The first feature should be the most important */
    assert(importances(1) == importances.max)
    /* They should all be non-zero */
    assert(importances.last == 0.0)

    assert(
      linearImportance.zip(importances).map { case (x, y) => x - y }.forall(d => Math.abs(d) < 1.0e-9),
      s"Expected linear and maxDepth=0 importances to align"
    )

    val result = DT.transform(trainingData.map(_.inputs))
    assert(result.depth.forall(_ == 0), s"Expected all the predictions to be depth 0")
  }

  /**
    * Test with random weights to ensure all feature importances of stump tree with linear leaves are non-negative
    */
  @Test
  def testWeights(): Unit = {
    val trainingData =
      DataGenerator.generate(32, 12, noise = 100.0, function = Friedman.friedmanSilverman, rng = rng).data
    val weightedData = trainingData.map(_.withWeight(rng.nextInt(8).toDouble))

    val linearLearner = LinearRegressionLearner(regParam = Some(1.0))
    val DTLearner = RegressionTreeLearner(leafLearner = Some(linearLearner), maxDepth = 1)
    val DTMeta = DTLearner.train(weightedData)

    /* The first feature should be the most important */
    val importances = DTMeta.featureImportance.get
    assert(importances.forall(_ >= 0.0), "Found negative feature importance")
  }

  /**
    * Test a simple tree with only real inputs
    */
  @Test
  def testSimpleBoltzmannTree(): Unit = {
    val csv = TestUtils.readCsv("double_example.csv")
    val trainingData = csv.map(vec => TrainingRow(vec.init, vec.last.asInstanceOf[Double], 1.0))
    val DTLearner = RegressionTreeLearner(splitter = BoltzmannSplitter(1e-4))
    val DT = DTLearner.train(trainingData, rng = rng).model

    /* We should be able to memorize the inputs */
    val output = DT.transform(trainingData.map(_.inputs))
    trainingData.zip(output.expected).foreach {
      case (row, p) => assert(Math.abs(row.label - p) < 1.0e-9)
    }
    assert(output.gradient.isEmpty)
    output.depth.zip(output.expected).foreach {
      case (d, y) => assert(d > 3 && d < 9, s"Depth is $d at y=$y")
    }
  }

  /**
    * Convenience method to compare the Shapley value at a test location for a tree trained on trainingData to a known reference.
    *
    * @param trainingData on which to train a regression tree
    * @param evalLocation at which to evaluate the Shapley value
    * @param expected     known reference Shapley value to which to compare (fails upon mismatch)
    */
  def shapleyCompare(
      trainingData: Seq[TrainingRow[Double]],
      evalLocation: Vector[Any],
      expected: Vector[Double],
      omitFeatures: Set[Int] = Set()
  ): Unit = {
    val actual = RegressionTreeLearner().train(trainingData).model.shapley(evalLocation, omitFeatures) match {
      case None => fail("Unexpected None returned by shapley.")
      case x: Option[DenseMatrix[Double]] => {
        val a = x.get
        assert(a.cols == trainingData.head.inputs.length, "Expected one Shapley value per feature.")
        assert(a.rows == 1, "Expected a single output dimension.")
        a.toDenseVector.toScalaVector
      }
      case _ => fail("Unexpected return type.")
    }
    assert(
      expected.zip(actual).forall {
        case (e: Double, a: Double) => Math.abs(e - a) < 1e-12
      },
      s"Shapley value ${actual} does not match reference ${expected}."
    )
  }

  /**
    * Test Shapley value for  a simple tree.
    */
  @Test
  def testShapley(): Unit = {
    // Example from Lundberg paper (https://arxiv.org/pdf/1802.03888.pdf)
    val trainingData1 = Seq(
      TrainingRow(Vector(1.0, 1.0), 80.0),
      TrainingRow(Vector(1.0, 0.0), 0.0),
      TrainingRow(Vector(0.0, 1.0), 0.0),
      TrainingRow(Vector(0.0, 0.0), 0.0)
    )
    val expected1 = Vector(30.0, 30.0)
    shapleyCompare(trainingData1, Vector[Any](1.0, 1.0), expected1)

    // Second example from Lundberg paper (https://arxiv.org/pdf/1802.03888.pdf)
    val trainingData2 = Seq(
      TrainingRow(Vector(1.0, 1.0), 90.0),
      TrainingRow(Vector(1.0, 0.0), 10.0),
      TrainingRow(Vector(0.0, 1.0), 0.0),
      TrainingRow(Vector(0.0, 0.0), 0.0)
    )
    val expected2 = Vector(35.0, 30.0)
    shapleyCompare(trainingData2, Vector[Any](1.0, 1.0), expected2)

    // Example with two splits on one feature
    // Worked out with pen-and-paper from Lundberg Equation 2.
    val trainingData3 = Seq(
      TrainingRow(Vector(1.0, 1.0), 100.0),
      TrainingRow(Vector(1.0, 0.0), 80.0),
      TrainingRow(Vector(1.0, 0.2), 70.0),
      TrainingRow(Vector(0.0, 1.0), 0.0),
      TrainingRow(Vector(0.0, 0.2), 0.0),
      TrainingRow(Vector(0.0, 0.0), 0.0)
    )
    val expected3 = Vector(45.8333333333333, 12.5)
    shapleyCompare(trainingData3, Vector[Any](1.0, 1.0), expected3)

    // Example with 5 features, to exercise all the factorials in Lundberg equation 2.
    // Referenced against the shap package on a sklearn decision tree.
    val trainingData4 = Seq(
      TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 0.0), 1.0),
      TrainingRow(Vector(1.0, 0.0, 0.0, 0.0, 0.0), 2.0),
      TrainingRow(Vector(0.0, 1.0, 0.0, 0.0, 0.0), 4.0),
      TrainingRow(Vector(0.0, 0.0, 1.0, 0.0, 0.0), 8.0),
      TrainingRow(Vector(0.0, 0.0, 0.0, 1.0, 0.0), 16.0),
      TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 1.0), 32.0)
    )
    val expected4 = Vector(0.0333333333333333, 0.2, 0.8666666666666667, 3.533333333333333, 16.866666666666667)
    shapleyCompare(trainingData4, Vector.fill[Any](5)(1.0), expected4)

    // Referenced against SHAP documentation TreeExplainer Single Split example
    val N = 100
    val M = 4
    val trainingData5 = Seq.fill(N / 2)(TrainingRow(Vector(1.0, 0.0, 0.0, 0.0, 0.0), 1.0)) ++
      Seq.fill(N / 2)(TrainingRow(Vector(0.0, 0.0, 0.0, 0.0, 0.0), 0.0))

    val expected5a = Vector(0.5, 0.0, 0.0, 0.0, 0.0)
    val expected5b = Vector(-0.5, 0.0, 0.0, 0.0, 0.0)
    shapleyCompare(trainingData5, Vector(1.0, 1.0, 1.0, 1.0, 1.0), expected5a)
    shapleyCompare(trainingData5, Vector(0.0, 0.0, 0.0, 0.0, 0.0), expected5b)

    // Referenced against SHAP documentation TreeExplainer Two feature AND example
    val trainingData6 = Seq.tabulate(N) { i =>
      val features = Vector
        .fill(M)(0.0)
        .updated(0, if (i < N / 2) 1.0 else 0.0)
        .updated(1, if (i < N / 4 || (i >= N / 2 && i < 3 * N / 4)) 1.0 else 0.0)
      val label = if (i < N / 4) 1.0 else 0.0
      TrainingRow(features, label)
    }

    val expected6a = Vector(0.375, 0.375, 0.0, 0.0, 0.0)
    val expected6b = Vector(-0.125, -0.125, 0.0, 0.0, 0.0)
    shapleyCompare(trainingData6, Vector(1.0, 1.0, 1.0, 1.0, 1.0), expected6a)
    shapleyCompare(trainingData6, Vector(0.0, 0.0, 0.0, 0.0, 0.0), expected6b)

    // Referenced against SHAP documentation TreeExplainer Two feature OR example
    val trainingData7 = Seq.tabulate(N) { i =>
      val features = Vector
        .fill(M)(0.0)
        .updated(0, if (i < N / 2) 1.0 else 0.0)
        .updated(1, if (i < N / 4 || (i >= N / 2 && i < 3 * N / 4)) 1.0 else 0.0)
      val label = if (i < N / 2 || (i >= N / 2 && i < 3 * N / 4)) 1.0 else 0.0
      TrainingRow(features, label)
    }

    val expected7a = Vector(0.125, 0.125, 0.0, 0.0)
    val expected7b = Vector(-0.375, -0.375, 0.0, 0.0)
    shapleyCompare(trainingData7, Vector(1.0, 1.0, 1.0, 1.0), expected7a)
    shapleyCompare(trainingData7, Vector(0.0, 0.0, 0.0, 0.0), expected7b)

    // Referenced against SHAP documentation TreeExplainer Two feature XOR example
    val trainingData8 = Seq.tabulate(N) { i =>
      val features = Vector
        .fill(M)(0.0)
        .updated(0, if (i < N / 2) 1.0 else 0.0)
        .updated(1, if (i < N / 4 || (i >= N / 2 && i < 3 * N / 4)) 1.0 else 0.0)
      val label = if ((i >= N / 4 && i < N / 2) || (i >= N / 2 && i < 3 * N / 4)) 1.0 else 0.0
      TrainingRow(features, label)
    }

    val expected8a = Vector(-0.25, -0.25, 0.0, 0.0)
    val expected8b = Vector(-0.25, -0.25, 0.0, 0.0)
    shapleyCompare(trainingData8, Vector(1.0, 1.0, 1.0, 1.0), expected8a)
    shapleyCompare(trainingData8, Vector(0.0, 0.0, 0.0, 0.0), expected8b)

    // Test omitted features
    val expected2a = Vector(0.0, 20.0)
    shapleyCompare(trainingData2, Vector[Any](1.0, 1.0), expected2a, omitFeatures = Set(0))
    val expected2b = Vector(25.0, 0.0)
    shapleyCompare(trainingData2, Vector[Any](1.0, 1.0), expected2b, omitFeatures = Set(1))

    // Ensure we don't crash when restricting number of features.
    RegressionTreeLearner(numFeatures = 1).train(trainingData4).model.shapley(Vector.fill[Any](5)(0.0), Set())
  }
}
