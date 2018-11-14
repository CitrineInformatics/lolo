from lolopy.learners import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_iris
from unittest import TestCase, main
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

        # Test with weights (make sure it doesn't crash)
        rf.fit(X, y, [1.0]*len(y))

        # Run predictions with std dev
        y_pred, y_std = rf.predict(X, return_std=True)
        self.assertEqual(len(y_pred), len(y_std))
        print('R^2:', r2_score(y_pred, y))

    def test_classifier(self):
        rf = RandomForestClassifier()

        # Load in the iris dataset
        X, y = load_iris(True)
        rf.fit(X, y)

        # Predict the probability of membership in each class
        y_prob = rf.predict_proba(X)
        self.assertEqual((len(X), 3), np.shape(y_prob))
        self.assertAlmostEqual(len(X), np.sum(y_prob))

        # Test just getting the predicted class
        y_pred = rf.predict(X)
        self.assertEqual(len(X), len(y_pred))
        print('Accuracy:', accuracy_score(y, y_pred))


if __name__ == "__main__":
    main()
