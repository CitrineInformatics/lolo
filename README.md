![Lolo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Rainy_Lake_in_Lolo_National_Forest.jpg/284px-Rainy_Lake_in_Lolo_National_Forest.jpg)

Lolo
====

![Travis](https://travis-ci.org/CitrineInformatics/lolo.svg?branch=develop)

Lolo is a [random forest](https://en.wikipedia.org/wiki/Lolo_National_Forest)-centered machine learning library in Scala.

The core of Lolo is bagging simple base learners, like decision trees, to imbue robust uncertainty estimates via 
[jackknife-style variance estimators](http://jmlr.org/papers/volume15/wager14a/wager14a.pdf) and explicit bias models.

Lolo supports:
 * continuous and categorical features
 * regression and classification trees
 * bagged learners to produce ensemble models, e.g. random forests
 * linear and ridge regression
 * regression _leaf models_, e.g. ridge regression trained on the leaf data
 * random rotation ensembles
 * bias-corrected jackknife-after-bootstrap and infinitesimal jackknife variance estimates
 * bias models trained on out-of-bag residuals
 * discrete influence scores, which characterize the response of a prediction each training instance
 * model based feature importance
 * distance correlation
 * hyperparameter optimization via grid or random search
 * out-of-bag error estimates
 * parallel training via scala parallel collections
 * validation metrics for accuracy and uncertainty quantification
 * visualization of predicted-vs-actual validations

# Usage
Lolo is on the central repository, and can be used by simply adding the following dependency block in your pom file:
```
<dependency>
    <groupId>io.citrine</groupId>
    <artifactId>lolo</artifactId>
    <version>3.0.1</version>
</dependency>
```
Lolo provides higher level wrappers for common learner combinations.
For example, you can use Random Forest with:
```
import io.citrine.lolo.learners.RandomForest
val trainingData: Seq[(Vector[Any], Any)] = features.zip(labels)
val model = new RandomForest().train(trainingData).getModel()
val predictions: Seq[Any] = model.transform(testInputs).getExpected()
```

# Performance
Lolo prioritizes functionality over performance, but it is still quite fast.  In its _random forest_ use case, the complexity scales as:

| Time complexity | Training rows | Features | Trees |
|-------|--------|-------|-------|
| `train` | O(n log n) | O(n) | O(n) |
| `getLoss` | O(n log n) | O(n) | O(n) |
| `getExpected` | O(log n) | O(1) | O(n) |
| `getUncertainty` | O(n) | O(1) | O(n) |

On an [Ivy Bridge](http://ark.intel.com/products/77780/Intel-Core-i7-4930K-Processor-12M-Cache-up-to-3_90-GHz) test platform, the (1024 row, 1024 tree, 8 feature) [performance test](src/test/scala/io/citrine/lolo/PerformanceTest.scala) took 1.4 sec to train and 2.3 ms per prediction with uncertainty.


# Contributing
We welcome bug reports, feature requests, and pull requests.  Pull requests should be made following the [gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).  As contributions expand, we'll put more information here.

# Authors
 * [Max Hutchinson](https://github.com/maxhutch/)
 * [Sean Paradiso](https://github.com/sparadiso)
 * [Logan Ward](https://github.com/WardLT)
 
# Related projects
 * [randomForestCI](https://github.com/swager/randomForestCI) is an R-based implementation of jackknife variance estimates by S. Wager
