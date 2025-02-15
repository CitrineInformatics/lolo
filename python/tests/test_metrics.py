from lolopy.metrics import root_mean_squared_error, standard_confidence, standard_error, uncertainty_correlation
from numpy.random import multivariate_normal, uniform, normal, seed

from pytest import approx

class TestMetrics:

    def test_rmse(self):
        assert root_mean_squared_error([1, 2], [1, 2]) == approx(0)
        assert root_mean_squared_error([4, 5], [1, 2]) == approx(3)

    def test_standard_confidence(self):
        assert standard_confidence([1, 2], [2, 3], [1.5, 0.9]) == approx(0.5)
        assert standard_confidence([1, 2], [2, 3], [1.5, 1.1]) == approx(1)

    def test_standard_error(self):
        assert standard_error([1, 2], [1, 2], [1, 1]) == approx(0)
        assert standard_error([4, 5], [1, 2], [3, 3]) == approx(1)

    def test_uncertainty_correlation(self):
        seed(3893789455)
        sample_size = 2 ** 15
        random_seed = 783245
        for expected in [0, 0.75]:
            # Make the error distribution
            y_true = uniform(0, 1, sample_size)

            # Make the errors and uncertainties
            draw = multivariate_normal([0, 0], [[1, expected], [expected, 1]], sample_size)

            # Add the errors, and separate out the standard deviations
            y_pred = y_true + [d[0] * normal(0, 1) for d in draw]
            y_std = [abs(d[1]) for d in draw]

            # Test with a very large tolerance for now
            measured_corr = uncertainty_correlation(y_true, y_pred, y_std, random_seed=random_seed)
            corr_error = abs(measured_corr - expected)
            assert corr_error < 0.25, f"Error for {expected:.2f}: {corr_error:.2f}"
