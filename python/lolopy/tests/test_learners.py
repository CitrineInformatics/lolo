from lolopy.learners import RandomForest
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from unittest import TestCase, main


class TestRF(TestCase):

    def test_rf_regressor(self):
        rf = RandomForest()

        # Train the model
        X, y = load_boston(True)
        rf.fit(X, y)

        # Run some predictions
        y_pred = rf.predict(X)
        self.assertEqual(len(y_pred), len(y))

        # Run predictions with std dev
        y_pred, y_std = rf.predict(X, return_std=True)
        self.assertEqual(len(y_pred), len(y_std))
        print('R^2:', r2_score(y_pred, y))


if __name__ == "__main__":
    main()
