![Lolo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Rainy_Lake_in_Lolo_National_Forest.jpg/284px-Rainy_Lake_in_Lolo_National_Forest.jpg)

Lolo
====

![Travis](https://travis-ci.org/CitrineInformatics/lolo.svg?branch=main)

Lolo is a [random forest](https://en.wikipedia.org/wiki/Lolo_National_Forest)-centered machine learning library in Scala.

The core of Lolo is bagging simple base learners, like decision trees, to produce models that can generate robust uncertainty estimates.

Lolo supports:
 * continuous and categorical features
 * regression, classification, and multi-task trees
 * bagged learners to produce ensemble models, e.g. random forests
 * linear and ridge regression
 * regression _leaf models_, e.g. ridge regression trained on the leaf data
 * random rotation ensembles
 * [recalibrated bootstrap prediction interval estimates](https://arxiv.org/abs/2205.02260)
 * bias-corrected jackknife-after-bootstrap and infinitesimal jackknife [confidence interval estimates](http://jmlr.org/papers/volume15/wager14a/wager14a.pdf)
 * bias models trained on out-of-bag residuals
 * feature importances computed via variance reduction or [Shapley values](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) (which are additive and per-prediction)
 * model based feature importance
 * distance correlation
 * hyperparameter optimization via grid or random search
 * parallel training via scala parallel collections
 * validation metrics for accuracy and uncertainty quantification
 * visualization of predicted-vs-actual validations
 * deterministic training via random seeds

# Usage
Lolo is on the central repository, and can be used by simply adding the following dependency block in your pom file:
```
<dependency>
    <groupId>io.citrine</groupId>
    <artifactId>lolo</artifactId>
    <version>6.0.0</version>
</dependency>
```
Lolo provides higher level wrappers for common learner combinations.
For example, you can use Random Forest with:
```
import io.citrine.lolo.learners.RandomForestRegressor
val trainingData: Seq[TrainingRow[Double]] = TrainingRow.build(features.zip(labels))
val model = RandomForestRegressor().train(trainingData).model
val predictions: Seq[Double] = model.transform(testInputs).expected
```

# Performance
Lolo prioritizes functionality over performance, but it is still quite fast.  In its _random forest_ use case, the complexity scales as:

| Time complexity | Training rows | Features | Trees |
|-----------------|--------|-------|-------|
| `train`         | O(n log n) | O(n) | O(n) |
| `loss`          | O(n log n) | O(n) | O(n) |
| `expected`      | O(log n) | O(1) | O(n) |
| `uncertainty`   | O(n) | O(1) | O(n) |

On an [Ivy Bridge](http://ark.intel.com/products/77780/Intel-Core-i7-4930K-Processor-12M-Cache-up-to-3_90-GHz) test platform, the (1024 row, 1024 tree, 8 feature) [performance test](src/test/scala/io/citrine/lolo/PerformanceTest.scala) took 1.4 sec to train and 2.3 ms per prediction with uncertainty.


# Contributing
We welcome bug reports, feature requests, and pull requests.
Pull requests should be made following the [feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow): branching off of and opening PRs into `main`.

Production releases are triggered by tags.
The [sbt-ci-release plugin](https://github.com/olafurpg/sbt-ci-release) will use the tag as the `lolo` version.
On the other hand, `lolopy` versions are still read from `setup.py`, so version bumps are needed for successful releases.
Failing to bump the `lolopy` version number will result in a skipped `lolopy` release rather than a build failure.

## Code Formatting
- Consistent formatting is enforced by scalafmt.
- The easiest way to check whether scalafmt is satisfied is to run scalafmt from the command line: `sbt scalafmtCheckAll`.
  This will check whether any files need to be reformatted.
  Pull requests are gated on this running successfully.
  You can automatically check whether code is formatted properly before pushing to an upstream repository using a git hook.
  To set this up, install the pre-commit framework by following the instructions [here](https://pre-commit.com/#installation).
  Then enable the hooks in `.pre-commit-config.yaml` by running `pre-commit install --hook-type pre-push` from the root directory.
  This will run `scalafmtCheckAll` before pushing to a remote repo.
- To ensure code is formatted properly, you can run `sbt scalafmtAll` from the command line or configure your IDE to format files on save.

# Authors

See [Contributors](https://github.com/CitrineInformatics/lolo/graphs/contributors)
 
# Related projects
 * [randomForestCI](https://github.com/swager/randomForestCI) is an R-based implementation of jackknife variance estimates by S. Wager
