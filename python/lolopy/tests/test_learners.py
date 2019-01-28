from lolopy.learners import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.datasets import load_boston, load_iris
from unittest import TestCase, main
import pickle as pkl
import numpy as np


class TestRF(TestCase):

    def test_rf_regressor(self):
        rf = RandomForestRegressor()

        # Train the model
        X, y = load_boston(True)
        rf.fit(X, y)

        # Run some predictions
        y_pred = rf.predict(X)
        self.assertEqual(len(y_pred), len(y))

        # Basic test for functionality. R^2 above 0.98 was measured on 27Dec18
        score = r2_score(y_pred, y)
        print('R^2:', score)
        self.assertGreater(score, 0.98)

        # Test with weights (make sure it doesn't crash)
        rf.fit(X, y, [1.0]*len(y))

        # Make sure feature importances are stored
        self.assertEqual(np.shape(rf.feature_importances_), (X.shape[1],))

        # Run predictions with std dev
        y_pred, y_std = rf.predict(X, return_std=True)
        self.assertEqual(len(y_pred), len(y_std))
        self.assertTrue((y_std >= 0).all())  # They must be positive
        self.assertGreater(np.std(y_std), 0)  # Must have a variety of values

        # Make sure the detach operation functions
        rf.clear_model()
        self.assertIsNone(rf.model_)

        # Test removing Jackknife, which should produce equal uncertainties for all entries
        rf.useJackknife = False
        rf.fit(X, y)
        y_pred, y_std = rf.predict(X, return_std=True)
        self.assertAlmostEqual(np.std(y_std), 0)

    def test_classifier(self):
        rf = RandomForestClassifier()

        # Load in the iris dataset
        X, y = load_iris(True)
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
        self.assertAlmostEqual(acc, 1)  # Given default settings, we should get perfect fittness to training data

    def test_serialization(self):
        rf = RandomForestClassifier()

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
