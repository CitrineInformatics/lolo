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
 * bias-corrected jackknife-after-bootstrap and infintessimal jackknife variance estimates
 * feature importance
 * out-of-bag error estimates
 * bias models trained on out-of-bag residuals
 * parallel training via Scala parallel collections
 * [experimental] jackknife-based training row scores

# Authors
 * [Max Hutchinson](https://github.com/maxhutch/)
 
# Related projects
 * [randomForestCI](https://github.com/swager/randomForestCI) is an R-based implementation of jackknife variance estimates by S. Wager
