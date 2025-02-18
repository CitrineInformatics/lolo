from lolopy.learners import (
    RandomForestRegressor,
    RandomForestClassifier,
    MultiTaskRandomForest,
    RegressionTreeLearner,
    LinearRegression,
    ExtraRandomTreesRegressor,
    ExtraRandomTreesClassifier
)
import numpy as np
import pickle
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.datasets import load_iris, load_diabetes, load_linnerud

from pytest import fixture, approx, raises
from numpy.testing import assert_allclose, assert_equal

@fixture(scope="function")
def linear_data():
    """Make data corresponding to y = x + 1

    Returns:
        np.ndarray: X
        np.ndarray: y
    """
    # Make y = x + 1
    X = np.linspace(0, 1, 32)
    y = X + 1
    # Make X a 2D array
    X = X[:, None]
    return X, y


class TestRF:

    def test_rf_regressor(self):
        rf = RandomForestRegressor()

        # Train the model
        X, y = load_diabetes(return_X_y=True)

        # Make sure we get a NotFittedError
        with raises(NotFittedError):
            rf.predict(X)

        # Verify y.shape is checked
        with raises(ValueError):
            rf.fit(X, y.reshape(y.shape[0], 1, 1))

        # Fit the model
        rf.fit(X, y, random_seed=31247895)

        # Run some predictions
        y_pred = rf.predict(X)
        assert y_pred.shape == y.shape

        # Test the ability to get importance scores
        y_import = rf.get_importance_scores(X[:100, :])
        assert y_import.shape == (100, len(X))

        # Basic test for functionality. R^2 above 0.88 was measured on 2021-12-09
        score = r2_score(y_pred, y)
        print('R^2:', score)
        assert score > 0.88

        # Test with weights (make sure it doesn't crash)
        rf.fit(X, y, [2.0]*len(y))

        # Make sure feature importances are stored
        assert rf.feature_importances_.shape == (X.shape[1],)
        assert rf.feature_importances_.sum() == approx(1.0)

        # Run predictions with std dev
        y_pred, y_std = rf.predict(X, return_std=True)
        assert y_pred.shape == y_std.shape
        assert (y_std >= 0).all()  # They must be positive
        assert y_std.std() > 0  # Must have a variety of values

        # For a single output, the covariance matrix is just the standard deviation squared
        _, y_cov = rf.predict(X, return_cov_matrix=True)
        assert_equal(y_cov.flatten(), y_std ** 2)

        # Make sure the detach operation functions
        rf.clear_model()
        assert rf.model_ is None


    def test_reproducibility(self):
        seed = 31247895
        rf1 = RandomForestRegressor()
        rf2 = RandomForestRegressor()
        X, y = load_diabetes(return_X_y=True)

        rf1.fit(X, y, random_seed=seed)
        rf2.fit(X, y, random_seed=seed)
        pred1 = rf1.predict(X)
        pred2 = rf2.predict(X)
        assert_equal(pred1, pred2)


    def test_rf_multioutput_regressor(self):
        rf = MultiTaskRandomForest()
        # A regression dataset with 3 outputs
        X, y = load_linnerud(return_X_y=True)
        num_data = len(X)
        num_outputs = y.shape[1]

        rf.fit(X, y, random_seed=810355)
        y_pred, y_std = rf.predict(X, return_std=True)
        _, y_cov = rf.predict(X, return_cov_matrix=True)

        # Assert that all returned values have the correct shape
        assert y_pred.shape == (num_data, num_outputs)
        assert y_std.shape == (num_data, num_outputs)
        assert y_cov.shape == (num_data, num_outputs, num_outputs)

        # The covariance matrices should be symmetric and the diagonals should be the squares of the standard deviations.
        assert_equal(y_cov[:, 0, 1], y_cov[:, 1, 0])
        assert_equal(y_cov[:, 0, 0], y_std[:, 0] ** 2)

        # Make sure the user cannot call predict with both return_std and return_cov_matrix True
        with raises(ValueError):
            rf.predict(X, return_std=True, return_cov_matrix=True)


    def test_classifier(self):
        rf = RandomForestClassifier()

        # Load in the iris dataset
        X, y = load_iris(return_X_y=True)
        rf.fit(X, y, random_seed=34789)

        # Predict the probability of membership in each class
        y_prob = rf.predict_proba(X)
        assert y_prob.shape == (len(X), 3)
        assert y_prob.sum() == approx(len(X))
        ll_score = log_loss(y, y_prob)
        print('Log loss:', ll_score)
        assert ll_score < 0.03  # Measured at 0.026 27Dec18

        # Test just getting the predicted class
        y_pred = rf.predict(X)
        assert_allclose(y_prob.argmax(axis=1), y_pred)
        assert len(y_pred) == len(X)
        acc = accuracy_score(y, y_pred)
        print('Accuracy:', acc)
        assert acc == approx(1)  # Given default settings, we should get perfect fitness to training data


    def test_regression_tree(self):
        tree = RegressionTreeLearner()

        # Make sure it trains and predicts properly
        X, y = load_diabetes(return_X_y=True)
        tree.fit(X, y)

        # Make sure the prediction works
        y_pred = tree.predict(X)

        # Full depth tree should yield perfect accuracy
        assert r2_score(y, y_pred) == approx(1)

        # Constrain tree depth severely
        tree.max_depth = 2
        tree.fit(X, y)
        y_pred = tree.predict(X)
        assert r2_score(y, y_pred) == approx(0.433370098)  # Result is deterministic

        # Constrain the tree to a single node, using minimum count per split
        tree = RegressionTreeLearner(min_leaf_instances=1000)
        tree.fit(X, y)
        assert r2_score(y, tree.predict(X)) == approx(0)


    def test_linear_regression(self, linear_data):
        lr = LinearRegression()

        # Make y = x + 1
        X, y = linear_data

        # Fit a linear regression model
        lr.fit(X, y)
        assert r2_score(y, lr.predict(X)) == 1

        # Not fitting an intercept
        lr.fit_intercept = False
        lr.fit(X, y)
        assert lr.predict([[0]])[0] == approx(0)

        # Add a regularization parameter, make sure the model fits
        lr.reg_param = 1
        lr.fit(X, y)


    def test_adjust_rtree_learners(self, linear_data):
        """Test modifying the bias and leaf learners of decision trees"""

        # Make a tree learner that will make only 1 split on 32 data points
        tree = RegressionTreeLearner(min_leaf_instances=16)

        # Make y = x + 1
        X, y = linear_data

        # Fit the model
        tree.fit(X, y)
        assert len(set(tree.predict(X))) == 2  # Only one split

        # Use linear regression on the splits
        tree.leaf_learner = LinearRegression()
        tree.fit(X, y)
        assert r2_score(y, tree.predict(X)) == approx(1.0)  # Linear leaves means perfect fit

        # Test whether changing leaf learner does something
        rf = RandomForestRegressor(leaf_learner=LinearRegression(), min_leaf_instances=16)
        rf.fit(X[:16, :], y[:16], random_seed=23478)  # Train only on a subset
        assert r2_score(y, rf.predict(X)) == approx(1.0)  # Should fit perfectly on whole dataset

        rf = RandomForestRegressor()
        rf.fit(X[:16, :], y[:16], random_seed=7834)
        assert r2_score(y, rf.predict(X)) < 1.0  # Should not fit the whole dataset perfectly


    def test_save_and_load_model(self, tmp_path, linear_data):
        file = tmp_path / 'test_model.json'

        rf = RandomForestRegressor(min_leaf_instances=16)

        # Load in the diabetes dataset
        X, y = linear_data
        rf.fit(X, y, random_seed=378456)

        # Save the model
        rf.save(file)

        # Load the model
        rf2 = RandomForestRegressor.load(file)

        # Make sure the predictions are the same
        pred1 = rf.predict(X)
        pred2 = rf2.predict(X)
        assert_equal(pred1, pred2)


    def test_pickle_model(self, tmp_path, linear_data):
        file = tmp_path / 'test_model.pkl'

        rf = RandomForestRegressor(min_leaf_instances=16)

        # Load in the diabetes dataset
        X, y = linear_data
        rf.fit(X, y, random_seed=378456)

        # Save the model
        with open(file, 'wb') as f:
            pickle.dump(rf, f)

        # Load the model
        with open(file, 'rb') as f:
            rf2 = pickle.load(f)

        # Make sure the predictions are the same
        pred1 = rf.predict(X)
        pred2 = rf2.predict(X)
        assert_equal(pred1, pred2)


class TestExtraRandomTrees:

    def test_extra_random_trees_regressor(self):
        # Setting disable_bootstrap=False allows us to generate uncertainty estimates from the bagged ensemble
        rf = ExtraRandomTreesRegressor(disable_bootstrap=False)

        # Train the model
        X, y = load_diabetes(return_X_y=True)

        # Make sure we get a NotFittedError
        with raises(NotFittedError):
            rf.predict(X)

        # Fit the model
        rf.fit(X, y, random_seed=378456)

        # Run some predictions
        y_pred = rf.predict(X)
        assert len(y_pred) == len(y)

        # Test the ability to get importance scores
        y_import = rf.get_importance_scores(X[:100, :])
        assert y_import.shape == (100, len(X))

        # Basic test for functionality. R^2 above 0.88 was measured on 2021-12-09
        score = r2_score(y_pred, y)
        print("R2: ", score)
        assert score > 0.88

        # Test with weights (make sure it doesn't crash)
        rf.fit(X, y, [2.0]*len(y))

        # Make sure feature importances are stored
        assert rf.feature_importances_.shape == (X.shape[1],)
        assert rf.feature_importances_.sum() == approx(1.0)

        # Run predictions with std dev
        # Requires disable_bootstrap=False on the learner to generate std dev alongside predictions
        y_pred, y_std = rf.predict(X, return_std=True)
        assert len(y_pred) == len(y_std)
        assert (y_std >= 0).all()  # They must be positive
        assert y_std.std() > 0  # Must have a variety of values

        # Make sure the detach operation functions
        rf.clear_model()
        assert rf.model_ is None


    def test_reproducibility(self):
        seed = 378456
        rf1 = ExtraRandomTreesRegressor()
        rf2 = ExtraRandomTreesRegressor()
        X, y = load_diabetes(return_X_y=True)

        rf1.fit(X, y, random_seed=seed)
        rf2.fit(X, y, random_seed=seed)
        pred1 = rf1.predict(X)
        pred2 = rf2.predict(X)
        assert_equal(pred1, pred2)


    def test_extra_random_trees_classifier(self):
        rf = ExtraRandomTreesClassifier()

        # Load in the iris dataset
        X, y = load_iris(return_X_y=True)
        rf.fit(X, y, random_seed=378456)

        # Predict the probability of membership in each class
        y_prob = rf.predict_proba(X)
        assert y_prob.shape == (len(X), 3)
        assert y_prob.sum() == approx(len(X))
        ll_score = log_loss(y, y_prob)
        print('Log loss:', ll_score)
        assert ll_score < 0.03  # Measured at 0.026 on 2020-04-06

        # Test just getting the predicted class
        y_pred = rf.predict(X)
        assert_allclose(y_prob.argmax(axis=1), y_pred)
        assert len(X) == len(y_pred)
        acc = accuracy_score(y, y_pred)
        print('Accuracy:', acc)
        assert acc == approx(1)  # Given default settings, we should get perfect fitness to training data
