![Lolo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Rainy_Lake_in_Lolo_National_Forest.jpg/284px-Rainy_Lake_in_Lolo_National_Forest.jpg)

Lolo
====

Lolo is a [random forest](https://en.wikipedia.org/wiki/Lolo_National_Forest)-centered machine learning library in Scala.

The core of Lolo is bagging simple base learners, like decision trees, to imbue robust uncertainty estimates via 
[jackknife-style variance estimators](http://www.jmlr.org/papers/volume15/wager14a/source/wager14a.pdf) and explicit bias models.

Lolo supports:
 * continuous and categorical features
 * regression and classification trees
 * bagged learners to produce ensemble models, e.g. random forests
 * linear and ridge regression
 * regression _leaf models_, e.g. ridge regression trained on the leaf data
 * bias-corrected jackknife-after-bootstrap and infintessimal jackknife variance estimates
 * feature importance
 * hyperparameter optimization via grid or random search
 * out-of-bag error estimates
 * bias models trained on out-of-bag residuals
 * parallel training via Scala parallel collections
 * [experimental] jackknife-based training row scores

# Usage
Lolo is not yet on maven central, so it needs to be installed manually:
```
git clone https://github.com/CitrineInformatics/lolo.git
cd lolo
mvn install
```
Lolo can then be used by adding the following dependency block in your pom file:
```
<dependency>
    <groupId>io.citrine</groupId>
    <artifactId>lolo</artifactId>
    <version>0.0.5</version>
</dependency>
```

# Contributing
We welcome bug reports, feature reqests, and pull requests.  Pull requests should be made following the [gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).  As contributions expand, we'll put more information here.

# Authors
 * [Max Hutchinson](https://github.com/maxhutch/)
 
# Related projects
 * [randomForestCI](https://github.com/swager/randomForestCI) is an R-based implementation of jackknife variance estimates by S. Wager
