# v3.0 is released!

Lolo version 3.0 is released! Notably, this release introduces a shiny new interface for uncertainty quantification that provides new and improved granular uncertainty quantification (UQ). This release also introduces two new random forest features, and improves the accuracy of predictions and UQ on certain types of machine learning problems.

### What’s New

* Now you can distinguish estimated uncertainty in observations from uncertainty in the mean! You can query each of these with a new interface that exposes granular estimates of the [bias-variance-noise decomposition](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). (#198)
* A new `Splitter` interface allowing custom tree splitting protocols such as the new `BoltzmannSplitter` used to train [Boltzmann trees](https://www.youtube.com/watch?v=wWChMOkNlWk). (#95)
* Support for [random rotation ensembles](http://www.jmlr.org/papers/v17/blaser16a.html), which is the technique of applying a different random rotation of feature space on each tree in a random forest. This can be helpful when a function to be learned aligns poorly to the coordinates of feature space. (#199 & #200)

### Improvements

* Bias-corrected jackknife variances are now rectified after summation instead of being rectified individually, improving the quality of importance scores. (#194)
* The implementation of BaggedResult has been simplified. (#195)

### Fixes

* When training data contained identical inputs associated with more than one categorical label, or when trees were not grown to full depth, GuessTheMeanLearner, ClassificationTreeLearner, and RandomForest were biased toward predicting labels that appeared earlier in the training dataset.
GuessTheMeanLearner now randomizes its tie-breaking choice of class label to predict. Consequently, GuessTheMeanLearner, ClassificationTreeLearner, and RandomForest are unbiased with respect to the order of training data and RandomForest performs substantially better on datasets with duplicated inputs. (#202)
* Release 3.0.0 lacked a rescaling factor and a square root in getStdDevObs. Version 3.0.1 fixes this, improves test coverage, and introduces a Bessel correction.
