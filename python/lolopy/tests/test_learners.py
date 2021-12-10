from lolopy.learners import (
    RandomForestRegressor,
    RandomForestClassifier,
    RegressionTreeLearner,
    LinearRegression,
    ExtraRandomTreesRegressor,
    ExtraRandomTreesClassifier
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.datasets import load_iris, load_diabetes, load_linnerud
from unittest import TestCase, main
import pickle as pkl
import numpy as np


def _make_linear_data():
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


class TestRF(TestCase):

    def test_rf_regressor(self):
        rf = RandomForestRegressor(random_seed = 31247895)

        # Train the model
        X, y = load_diabetes(return_X_y=True)

        # Make sure we get a NotFittedError
        with self.assertRaises(NotFittedError):
            rf.predict(X)

        # Fit the model
        rf.fit(X, y)

        # Run some predictions
        y_pred = rf.predict(X)
        self.assertEqual(len(y_pred), len(y))

        # Test the ability to get importance scores
        y_import = rf.get_importance_scores(X[:100, :])
        self.assertEqual((100, len(X)), y_import.shape)

        # Basic test for functionality. R^2 above 0.88 was measured on 2021-12-09
        score = r2_score(y_pred, y)
        print('R^2:', score)
        self.assertGreater(score, 0.88)

        # Test with weights (make sure it doesn't crash)
        rf.fit(X, y, [2.0]*len(y))

        # Make sure feature importances are stored
        self.assertEqual(np.shape(rf.feature_importances_), (X.shape[1],))
        self.assertAlmostEqual(1.0, np.sum(rf.feature_importances_))

        # Run predictions with std dev
        y_pred, y_std = rf.predict(X, return_std=True)
        self.assertEqual(len(y_pred), len(y_std))
        self.assertTrue((y_std >= 0).all())  # They must be positive
        self.assertGreater(np.std(y_std), 0)  # Must have a variety of values

        # For a single output, the covariance matrix is just the standard deviation squared
        _, y_cov = rf.predict(X, return_cov_matrix=True)
        assert np.all(y_cov.flatten() == y_std ** 2)

        # Make sure the detach operation functions
        rf.clear_model()
        self.assertIsNone(rf.model_)

    def test_rf_multioutput_regressor(self):
        rf = RandomForestRegressor(random_seed=810355)
        # A regression dataset with 3 outputs
        X, y = load_linnerud(return_X_y=True)
        num_data = len(X)
        num_outputs = y.shape[1]

        rf.fit(X, y)
        y_pred, y_std = rf.predict(X, return_std=True)
        _, y_cov = rf.predict(X, return_cov_matrix=True)

        # Assert that all returned values have the correct shape
        assert y_pred.shape == (num_data, num_outputs)
        assert y_std.shape == (num_data, num_outputs)
        assert y_cov.shape == (num_data, num_outputs, num_outputs)

        # The covariance matrices should be symmetric and the diagonals should be the squares of the standard deviations.
        assert np.all(y_cov[:, 0, 1] == y_cov[:, 1, 0])
        assert np.all(y_cov[:, 0, 0] == y_std[:, 0] ** 2)

        # Make sure the user cannot call predict with both return_std and return_cov_matrix True
        with self.assertRaises(ValueError):
            rf.predict(X, return_std=True, return_cov_matrix=True)

    def test_classifier(self):
        rf = RandomForestClassifier(random_seed = 34789)

        # Load in the iris dataset
        X, y = load_iris(return_X_y=True)
        rf.fit(X, y)

        # Predict the probability of membership in each class
        y_prob = rf.predict_proba(X)
        self.assertEqual((len(X), 3), np.shape(y_prob))
        self.assertAlmostEqual(len(X), np.sum(y_prob))
        ll_score = log_loss(y, y_prob)
        print('Log loss:', ll_score)
        self.assertLess(ll_score, 0.03)  # Measured at 0.026 27Dec18

        # Test just getting the predicted class
        y_pred = rf.predict(X)
        self.assertTrue(np.isclose(np.argmax(y_prob, 1), y_pred).all())
        self.assertEqual(len(X), len(y_pred))
        acc = accuracy_score(y, y_pred)
        print('Accuracy:', acc)
        self.assertAlmostEqual(acc, 1)  # Given default settings, we should get perfect fitness to training data

    def test_serialization(self):
        rf = RandomForestClassifier(random_seed = 234785)

        # Make sure this doesn't error without a model training
        data = pkl.dumps(rf)
        rf2 = pkl.loads(data)

        # Load in the iris dataset and train model
        X, y = load_iris(return_X_y=True)
        rf.fit(X, y)

        # Try saving and loading the model
        data = pkl.dumps(rf)
        rf2 = pkl.loads(data)

        # Make sure it yields the same predictions as the first model
        probs1 = rf.predict_proba(X)
        probs2 = rf2.predict_proba(X)
        self.assertTrue(np.isclose(probs1, probs2).all())

    def test_regression_tree(self):
        tree = RegressionTreeLearner()

        # Make sure it trains and predicts properly
        X, y = load_diabetes(return_X_y=True)
        tree.fit(X, y)

        # Make sure the prediction works
        y_pred = tree.predict(X)

        # Full depth tree should yield perfect accuracy
        self.assertAlmostEqual(1, r2_score(y, y_pred))

        # Constrain tree depth severely
        tree.max_depth = 2
        tree.fit(X, y)
        y_pred = tree.predict(X)
        self.assertAlmostEqual(0.433370098, r2_score(y, y_pred))  # Result is deterministic

        # Constrain the tree to a single node, using minimum count per split
        tree = RegressionTreeLearner(min_leaf_instances=1000)
        tree.fit(X, y)
        self.assertAlmostEqual(0, r2_score(y, tree.predict(X)))

    def test_linear_regression(self):
        lr = LinearRegression()

        # Make y = x + 1
        X, y = _make_linear_data()

        # Fit a linear regression model
        lr.fit(X, y)
        self.assertEqual(1, r2_score(y, lr.predict(X)))

        # Not fitting an intercept
        lr.fit_intercept = False
        lr.fit(X, y)
        self.assertAlmostEqual(0, lr.predict([[0]])[0])

        # Add a regularization parameter, make sure the model fits
        lr.reg_param = 1
        lr.fit(X, y)

    def test_adjust_rtree_learners(self):
        """Test modifying the bias and leaf learners of decision trees"""

        # Make a tree learner that will make only 1 split on 32 data points
        tree = RegressionTreeLearner(min_leaf_instances=16)

        # Make y = x + 1
        X, y = _make_linear_data()

        # Fit the model
        tree.fit(X, y)
        self.assertEqual(2, len(set(tree.predict(X))))  # Only one split

        # Use linear regression on the splits
        tree.leaf_learner = LinearRegression()
        tree.fit(X, y)
        self.assertAlmostEqual(1.0, r2_score(y, tree.predict(X)))  # Linear leaves means perfect fit

        # Test whether changing leaf learner does something
        rf = RandomForestRegressor(leaf_learner=LinearRegression(), min_leaf_instances=16, random_seed = 23478)
        rf.fit(X[:16, :], y[:16])  # Train only on a subset
        self.assertAlmostEqual(1.0, r2_score(y, rf.predict(X)))  # Should fit perfectly on whole dataset

        rf = RandomForestRegressor(random_seed = 7834)
        rf.fit(X[:16, :], y[:16])
        self.assertLess(r2_score(y, rf.predict(X)), 1.0)  # Should not fit the whole dataset perfectly


class TestExtraRandomTrees(TestCase):

    def test_extra_random_trees_regressor(self):
        rf = ExtraRandomTreesRegressor(random_seed = 378456)

        # Train the model
        X, y = load_diabetes(return_X_y=True)

        # Make sure we get a NotFittedError
        with self.assertRaises(NotFittedError):
            rf.predict(X)

        # Fit the model
        rf.fit(X, y)

        # Run some predictions
        y_pred = rf.predict(X)
        self.assertEqual(len(y_pred), len(y))

        # Test the ability to get importance scores
        y_import = rf.get_importance_scores(X[:100, :])
        self.assertEqual((100, len(X)), y_import.shape)

        # Basic test for functionality. R^2 above 0.88 was measured on 2021-12-09
        score = r2_score(y_pred, y)
        print("R2: ", score)
        self.assertGreater(score, 0.88)

        # Test with weights (make sure it doesn't crash)
        rf.fit(X, y, [2.0]*len(y))

        # Make sure feature importances are stored
        self.assertEqual(np.shape(rf.feature_importances_), (X.shape[1],))
        self.assertAlmostEqual(1.0, np.sum(rf.feature_importances_))

        # Run predictions with std dev
        y_pred, y_std = rf.predict(X, return_std=True)
        self.assertEqual(len(y_pred), len(y_std))
        self.assertTrue((y_std >= 0).all())  # They must be positive
        self.assertGreater(np.std(y_std), 0)  # Must have a variety of values

        # Make sure the detach operation functions
        rf.clear_model()
        self.assertIsNone(rf.model_)

    def test_extra_random_trees_classifier(self):
        rf = ExtraRandomTreesClassifier(random_seed = 378456)

        # Load in the iris dataset
        X, y = load_iris(True)
        rf.fit(X, y)

        # Predict the probability of membership in each class
        y_prob = rf.predict_proba(X)
        self.assertEqual((len(X), 3), np.shape(y_prob))
        self.assertAlmostEqual(len(X), np.sum(y_prob))
        ll_score = log_loss(y, y_prob)
        print('Log loss:', ll_score)
        self.assertLess(ll_score, 0.03)  # Measured at 0.026 on 2020-04-06

        # Test just getting the predicted class
        y_pred = rf.predict(X)
        self.assertTrue(np.isclose(np.argmax(y_prob, 1), y_pred).all())
        self.assertEqual(len(X), len(y_pred))
        acc = accuracy_score(y, y_pred)
        print('Accuracy:', acc)
        self.assertAlmostEqual(acc, 1)  # Given default settings, we should get perfect fitness to training data

    def test_serialization(self):
        rf = ExtraRandomTreesClassifier(random_seed = 378945)

        # Make sure this doesn't error without a model training
        data = pkl.dumps(rf)
        rf2 = pkl.loads(data)

        # Load in the iris dataset and train model
        X, y = load_iris(True)
        rf.fit(X, y)

        # Try saving and loading the model
        data = pkl.dumps(rf)
        rf2 = pkl.loads(data)

        # Make sure it yields the same predictions as the first model
        probs1 = rf.predict_proba(X)
        probs2 = rf2.predict_proba(X)
        self.assertTrue(np.isclose(probs1, probs2).all())


if __name__ == "__main__":
    main()
