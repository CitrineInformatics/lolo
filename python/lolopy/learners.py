from abc import abstractmethod, ABCMeta

import sys
import numpy as np
from lolopy.loloserver import get_java_gateway
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_regressor

__all__ = ['RandomForestRegressor', 'RandomForestClassifier']


class BaseLoloLearner(BaseEstimator, metaclass=ABCMeta):
    """Base object for all leaners that use Lolo.

    Contains logic for starting the JVM gateway, and the fit operations.
    It is only necessary to implement the `_make_learner` object and create an `__init__` function
    to adapt a learner from the Lolo library for use in lolopy.

    The logic for making predictions (i.e., `predict` and `predict_proba`) is specific to whether the learner
    is a classification or regression model.
    In lolo, learners are not specific to a regression or classification problem and the type of problem is determined
    when fitting data is provided to the algorithm.
    In contrast, Scikit-learn learners for regression or classification problems are different classes.
    We have implemented `BaseLoloRegressor` and `BaseLoloClassifier` abstract classes to make it easier for creating
    a classification or regression version of a Lolo base class.
    The pattern for creating a scikit-learn compatible learner is to first implement the `_make_learner` and `__init__`
    operations in a special "Mixin" class that inherits from `BaseLoloLearner`, and then create a regression- or
    classification-specific class that inherits from both `BaseClassifier` or `BaseRegressor` and your new "Mixin".
    See the RandomForest models as an example for this approach.
    """

    def __init__(self):
        self.gateway = get_java_gateway()

        # Create a placeholder for the model
        self.model_ = None
        self._compress_level = 9
        
    def __getstate__(self):
        # Get the current state
        try:
            state = super(BaseLoloLearner, self).__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        # Delete the gateway data
        del state['gateway']

        # If there is a model set, replace it with the JVM copy
        if self.model_ is not None:
            state['model_'] = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.serializeObject(self.model_,
                                                                                                     self._compress_level)
        return state

    def __setstate__(self, state):
        # Unpickle the object
        super(BaseLoloLearner, self).__setstate__(state)

        # Get a pointer to the gateway
        self.gateway = get_java_gateway()

        # If needed, load the model into memory
        if state['model_'] is not None:
            bytes = state.pop('model_')
            self.model_ = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.deserializeObject(bytes)

    def fit(self, X, y, weights=None):
        # Instantiate the JVM object
        learner = self._make_learner()

        # Convert all of the training data to Java arrays
        train_data, weights_java = self._convert_train_data(X, y, weights)
        assert train_data.length() == len(X), "Array copy failed"
        assert train_data.head()._1().length() == len(X[0]), "Wrong number of features"
        assert weights_java.length() == len(X), "Weights copy failed"

        # Train the model
        result = learner.train(train_data, self.gateway.jvm.scala.Some(weights_java))

        # Unlink the training data, which is no longer needed (to save memory)
        self.gateway.detach(train_data)
        self.gateway.detach(weights_java)

        # Get the model out
        self.model_ = result.getModel()

        return self

    @abstractmethod
    def _make_learner(self):
        """Instantiate the learner used by Lolo to train a model

        Returns:
            (JavaObject) A lolo "Learner" object, which can be used to train a model"""
        pass

    def clear_model(self):
        """Utility operation for deleting model from JVM when no longer needed"""

        if self.model_ is not None:
            self.gateway.detach(self.model_)
            self.model_ = None

    def _convert_train_data(self, X, y, weights=None):
        """Convert the training data to a form accepted by Lolo

        Args:
            X (ndarray): Input variables
            y (ndarray): Output variables
            weights (ndarray): Wegihts for each sample
        Returns
            train_data (JavaObject): Pointer to the training data in Java
        """

        # Make some default weights
        if weights is None:
            weights = np.ones(len(y))

        # Convert x, y, and w to float64 and int8 with native ordering
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64 if is_regressor(self) else np.int32)
        weights = np.array(weights, dtype=np.float64)
        big_end = sys.byteorder == "big"

        # Convert X and y to Java Objects
        X_java = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getFeatureArray(X.tobytes(), X.shape[1], big_end)
        y_java = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.get1DArray(y.tobytes(), is_regressor(self), big_end)
        assert y_java.length() == len(y) == len(X)
        w_java = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.get1DArray(np.array(weights).tobytes(), True, big_end)
        assert w_java.length() == len(weights)

        return self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.zipTrainingData(X_java, y_java), w_java

    def _convert_run_data(self, X):
        """Convert the data to be run by the model

        Args:
            X (ndarray): Input data
        Returns:
            (JavaObject): Pointer to run data in Java
        """

        return self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getFeatureArray(X.tobytes(), X.shape[1], False)


class BaseLoloRegressor(BaseLoloLearner, RegressorMixin):
    """Abstract class for models that produce regression models.

    Implements the predict operation"""

    def predict(self, X, return_std=False):
        # Convert the data to Java
        X_java = self._convert_run_data(X)

        # Get the PredictionResult
        pred_result = self.model_.transform(X_java)

        # Unlink the run data, which is no longer needed (to save memory)
        self.gateway.detach(X_java)

        # Pull out the expected values
        y_pred_byte = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getRegressionExpected(pred_result)
        y_pred = np.frombuffer(y_pred_byte, dtype='float')  # Lolo gives a byte array back

        # If desired, return the uncertainty too
        if return_std:
            # TODO: This part fails on Windows because the NativeSystemBLAS is not found. Fix that
            # TODO: This is only valid for regression models. Perhaps make a "LoloRegressor" class
            y_std_bytes = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getRegressionUncertainty(pred_result)
            y_std = np.frombuffer(y_std_bytes, 'float')
            return y_pred, y_std

        # Get the expected values
        return y_pred


class BaseLoloClassifier(BaseLoloLearner, ClassifierMixin):
    """Base class for classification models

    Implements a modification to the fit operation that stores the number of classes
    and the predict/predict_proba methods"""

    def fit(self, X, y, weights=None):
        # Get the number of classes
        self.n_classes_ = len(set(y))

        return super(BaseLoloClassifier, self).fit(X, y, weights)

    def predict(self, X):
        # Convert the data to Java
        X_java = self._convert_run_data(X)

        # Get the PredictionResult
        pred_result = self.model_.transform(X_java)

        # Unlink the run data, which is no longer needed (to save memory)
        self.gateway.detach(X_java)

        # Pull out the expected values
        y_pred_byte = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getClassifierExpected(pred_result)
        y_pred = np.frombuffer(y_pred_byte, dtype=np.int32)  # Lolo gives a byte array back

        return y_pred

    def predict_proba(self, X):
        # Convert the data to Java
        X_java = self._convert_run_data(X)

        # Get the PredictionResult
        pred_result = self.model_.transform(X_java)

        # Unlink the run data, which is no longer needed (to save memory)
        self.gateway.detach(X_java)

        # Copy over the class probabilities
        probs_byte = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getClassifierProbabilities(pred_result,
                                                                                                   self.n_classes_)
        probs = np.frombuffer(probs_byte, dtype='float').reshape(-1, self.n_classes_)
        return probs


class RandomForestMixin(BaseLoloLearner):
    """Random Forest base class

    Implements the _make_learner operation and the __init__ function with options specific to the RandomForest
    class in Lolo"""

    def __init__(self, num_trees=-1, useJackknife=True, subsetStrategy="auto"):
        """Initialize the RandomForest

        Args:
            num_trees (int): Number of trees to use in the forest
        """
        super(RandomForestMixin, self).__init__()

        # Get JVM for this object

        # Store the variables
        self.num_trees = num_trees
        self.useJackknife = useJackknife
        self.subsetStrategy = subsetStrategy

    def _make_learner(self):
        #  TODO: Figure our a more succinct way of dealing with optional arguments/Option values
        #  TODO: Do not hard-code use of RandomForest
        learner = self.gateway.jvm.io.citrine.lolo.learners.RandomForest(
            self.num_trees, self.useJackknife,
            getattr(self.gateway.jvm.io.citrine.lolo.learners.RandomForest,
                    "$lessinit$greater$default$3")(),
            getattr(self.gateway.jvm.io.citrine.lolo.learners.RandomForest,
                    "$lessinit$greater$default$4")(),
            self.subsetStrategy
        )
        return learner


class RandomForestRegressor(BaseLoloRegressor, RandomForestMixin):
    """Random Forest model used for regression"""


class RandomForestClassifier(BaseLoloClassifier, RandomForestMixin):
    """Random Forest model used for classiciation"""
