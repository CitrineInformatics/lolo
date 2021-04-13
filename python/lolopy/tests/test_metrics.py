from lolopy.loloserver import get_java_gateway
from lolopy.metrics import (root_mean_squared_error, standard_confidence, standard_error, uncertainty_correlation)
from numpy.random import multivariate_normal, uniform, normal, seed
from unittest import TestCase


class TestMetrics(TestCase):

    def test_rmse(self):
        self.assertAlmostEqual(root_mean_squared_error([1, 2], [1, 2]), 0)
        self.assertAlmostEqual(root_mean_squared_error([4, 5], [1, 2]), 3)

    def test_standard_confidene(self):
        gateway = get_java_gateway()
        rng = gateway.jvm.scala.util.Random(367894)
        self.assertAlmostEqual(standard_confidence([1, 2], [2, 3], [1.5, 0.9]), 0.5, rng)
        self.assertAlmostEqual(standard_confidence([1, 2], [2, 3], [1.5, 1.1]), 1, rng)

    def test_standard_error(self):
        self.assertAlmostEqual(standard_error([1, 2], [1, 2], [1, 1]), 0)
        self.assertAlmostEqual(standard_error([4, 5], [1, 2], [3, 3]), 1)

    def test_uncertainty_correlation(self):
        seed(3893789455)
        sample_size = 2 ** 15
        gateway = get_java_gateway()
        rng = gateway.jvm.scala.util.Random(783245)
        for expected in [0, 0.75]:
            # Make the error distribution
            y_true = uniform(0, 1, sample_size)

            # Make the errors and uncertainties
            draw = multivariate_normal([0, 0], [[1, expected], [expected, 1]], sample_size)

            # Add the errors, and separate out the standard deviations
            y_pred = y_true + [d[0] * normal(0, 1) for d in draw]
            y_std = [abs(d[1]) for d in draw]

            # Test with a very large tolerance for now
            measured_corr = uncertainty_correlation(y_true, y_pred, y_std, rng = rng)
            corr_error = abs(measured_corr - expected)
            self.assertLess(corr_error, 0.25, 'Error for {:.2f}: {:.2f}'.format(expected, corr_error))
