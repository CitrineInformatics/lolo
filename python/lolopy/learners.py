from abc import abstractmethod, ABCMeta

import numpy as np
from lolopy.loloserver import get_java_gateway
from lolopy.utils import send_feature_array, send_1D_array
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_regressor
from sklearn.exceptions import NotFittedError

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
        self.feature_importances_ = None
        
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

        # Store the feature importances
        feature_importances_java = result.getFeatureImportance().get()
        feature_importances_bytes = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.send1DArray(feature_importances_java)
        self.feature_importances_ = np.frombuffer(feature_importances_bytes, 'float')

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

        y = np.array(y, dtype=np.float64 if is_regressor(self) else np.int32)
        weights = np.array(weights, dtype=np.float64)


        # Convert X and y to Java Objects
        X_java = send_feature_array(self.gateway, X)
        y_java = send_1D_array(self.gateway, y, is_regressor(self))
        assert y_java.length() == len(y) == len(X)
        w_java = send_1D_array(self.gateway, weights, True)
        assert w_java.length() == len(weights)

        return self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.zipTrainingData(X_java, y_java), w_java

    def _convert_run_data(self, X):
        """Convert the data to be run by the model

        Args:
            X (ndarray): Input data
        Returns:
            (JavaObject): Pointer to run data in Java
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getFeatureArray(X.tobytes(), X.shape[1], False)

    def get_importance_scores(self, X):
        """Get the importance scores for each entry in the training set for each prediction
        
        Args:
            X (ndarray): Inputs for each entry to be assessed
        """

        pred_result = self._get_prediction_result(X)

        y_import_bytes = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getImportanceScores(pred_result)
        y_import = np.frombuffer(y_import_bytes, 'float').reshape(len(X), -1)
        return y_import

    def _get_prediction_result(self, X):
        """Get the PredictionResult from the lolo JVM
        
        The PredictionResult class holds methods that will generate the expected predictions, uncertainty intervals, etc
        
        Args:
            X (ndarray): Input features for each entry
        Returns:
            (JavaObject): Prediction result produced by evaluating the model
        """

        # Check that the model is fitted
        if self.model_ is None:
            raise NotFittedError()

        # Convert the data to Java
        X_java = self._convert_run_data(X)

        # Get the PredictionResult
        pred_result = self.model_.transform(X_java)

        # Unlink the run data, which is no longer needed (to save memory)
        self.gateway.detach(X_java)
        return pred_result


class BaseLoloRegressor(BaseLoloLearner, RegressorMixin):
    """Abstract class for models that produce regression models.

    Implements the predict operation"""

    def predict(self, X, return_std=False):
        # Start the prediction process
        pred_result = self._get_prediction_result(X)

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
        pred_result = self._get_prediction_result(X)

        # Pull out the expected values
        y_pred_byte = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getClassifierExpected(pred_result)
        y_pred = np.frombuffer(y_pred_byte, dtype=np.int32)  # Lolo gives a byte array back

        return y_pred

    def predict_proba(self, X):
        pred_result = self._get_prediction_result(X)

        # Copy over the class probabilities
        probs_byte = self.gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getClassifierProbabilities(pred_result,
                                                                                                   self.n_classes_)
        probs = np.frombuffer(probs_byte, dtype='float').reshape(-1, self.n_classes_)
        return probs


class RandomForestMixin(BaseLoloLearner):
    """Random Forest base class

    Implements the _make_learner operation and the __init__ function with options specific to the RandomForest
    class in Lolo"""

    def __init__(self, num_trees=-1, use_jackknife=True, bias_learner=None,
                 leaf_learner=None, subset_strategy="auto", min_leaf_instances=1,
                 max_depth=2**30, uncertainty_calibration=False, randomize_pivot_location=False,
                 randomly_rotate_features=False):
        """Initialize the RandomForest

        Args:
            num_trees (int): Number of trees to use in the forest
            use_jackknife (bool): Whether to use jackknife based variance estimates
            bias_learner (BaseLoloLearner): Algorithm used to model bias (default: no model)
            leaf_learner (BaseLoloLearner): Learner used at each leaf of the random forest (default: GuessTheMean)
            subset_strategy (Union[string,int,float]): Strategy used to determine number of features used at each split 
                Available options:
                    "auto": Use the default for lolo (all features for regression, sqrt for classification)
                    "log2": Use the base 2 log of the number of features
                    "sqrt": Use the square root of the number of features
                    integer: Set the number of features explicitly
                    float: Use a certain fraction of the features
            min_leaf_instances (int): Minimum number of features used at each leaf
            max_depth (int): Maximum depth to which to allow the decision trees to grow
            uncertainty_calibration (bool): whether to re-calibrate the predicted uncertainty based on out-of-bag residuals
            randomize_pivot_location (bool): whether to draw pivots randomly or always select the midpoint
            randomly_rotate_features (bool): whether to randomly rotate real features for each tree in the forest
        """
        super(RandomForestMixin, self).__init__()

        # Store the variables
        self.num_trees = num_trees
        self.use_jackknife = use_jackknife
        self.subset_strategy = subset_strategy
        self.bias_learner = bias_learner
        self.leaf_learner = leaf_learner
        self.min_leaf_instances = min_leaf_instances
        self.max_depth = max_depth
        self.uncertainty_calibration = uncertainty_calibration
        self.randomize_pivot_location = randomize_pivot_location
        self.randomly_rotate_features = randomly_rotate_features

    def _make_learner(self):
        #  TODO: Figure our a more succinct way of dealing with optional arguments/Option values
        #  TODO: that ^^, please
        learner = self.gateway.jvm.io.citrine.lolo.learners.RandomForest(
            self.num_trees, self.use_jackknife,
            getattr(self.gateway.jvm.io.citrine.lolo.learners.RandomForest,
                    "$lessinit$greater$default$3")() if self.bias_learner is None
            else self.gateway.jvm.scala.Some(self.bias_learner._make_learner()),
            getattr(self.gateway.jvm.io.citrine.lolo.learners.RandomForest,
                    "$lessinit$greater$default$4")() if self.leaf_learner is None
            else self.gateway.jvm.scala.Some(self.leaf_learner._make_learner()),
            self.subset_strategy,
            self.min_leaf_instances,
            self.max_depth,
            self.uncertainty_calibration,
            self.randomize_pivot_location,
            self.randomly_rotate_features
        )
        return learner


class RandomForestRegressor(BaseLoloRegressor, RandomForestMixin):
    """Random Forest model used for regression"""


class RandomForestClassifier(BaseLoloClassifier, RandomForestMixin):
    """Random Forest model used for classification"""


class RegressionTreeLearner(BaseLoloRegressor):
    """Regression tree learner, based on the RandomTree algorithm."""

    def __init__(self, num_features=-1, max_depth=30, min_leaf_instances=1, leaf_learner=None):
        """Initialize the learner

        Args:
            num_features (int): Number of features to consider at each split (-1 to consider all features)
            max_depth (int): Maximum depth of the regression tree
            min_leaf_instances (int): Minimum number instances per leaf
            leaf_learner (BaseLoloLearner): Learner to use on the leaves
        """
        super(RegressionTreeLearner, self).__init__()
        self.num_features = num_features
        self.max_depth = max_depth
        self.min_leaf_instances = min_leaf_instances
        self.leaf_learner = leaf_learner

    def _make_learner(self):
        if self.leaf_learner is None:
            # pull out default learner
            leaf_learner = getattr(
                self.gateway.jvm.io.citrine.lolo.trees.regression.RegressionTreeLearner,
                "$lessinit$greater$default$4"
            )()
        else:
            leaf_learner = self.gateway.jvm.scala.Some(self.leaf_learner._make_learner())

        # pull out default splitter
        splitter = getattr(
            self.gateway.jvm.io.citrine.lolo.trees.regression.RegressionTreeLearner,
            "$lessinit$greater$default$5"
        )()

        return self.gateway.jvm.io.citrine.lolo.trees.regression.RegressionTreeLearner(
            self.num_features, self.max_depth, self.min_leaf_instances,
            leaf_learner, splitter
        )

class LinearRegression(BaseLoloRegressor):
    """Linear ridge regression with an :math:`L_2` penalty"""

    def __init__(self, reg_param=None, fit_intercept=True):
        """Initialize the regressor"""

        super(LinearRegression, self).__init__()

        self.reg_param = reg_param
        self.fit_intercept = fit_intercept

    def _make_learner(self):
        return self.gateway.jvm.io.citrine.lolo.linear.LinearRegressionLearner(
            getattr(self.gateway.jvm.io.citrine.lolo.linear.LinearRegressionLearner,
                    "$lessinit$greater$default$1")() if self.reg_param is None
            else self.gateway.jvm.scala.Some(float(self.reg_param)),
            self.fit_intercept
        )
