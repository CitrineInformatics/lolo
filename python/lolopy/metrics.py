"""Functions to call lolo Merit classes, which describe the performance of a machine learning model

These are very similar to the "metrics" from scikit-learn, which is the reason for the name of this module.
"""

from lolopy.loloserver import get_java_gateway
from lolopy.utils import send_1D_array
import numpy as np


def _call_lolo_merit(metric_name, y_true, y_pred, random_seed=None, y_std=None, *args):
    """Call a metric from lolopy
    
    Args:
        metric_name (str): Name of a Merit class (e.g., UncertaintyCorrelation)
        y_true ([double]): True value
        y_pred ([double]): Predicted values
        random_seed (int): for reproducibility (only used by some metrics)
        y_std ([double]): Prediction uncertainties (only used by some metrics)
        *args: Any parameters to the constructor of the Metric
    Returns:
        (double): Metric score
    """

    # If needed, set y_std to 1 for all entries
    if y_std is None:
        y_std = np.ones(len(y_true))

    gateway = get_java_gateway()
    # Get default rng
    rng = gateway.jvm.io.citrine.lolo.util.LoloPyRandom.getRng(random_seed) if random_seed \
        else gateway.jvm.io.citrine.lolo.util.LoloPyRandom.getRng()
    # Get the metric object
    metric = getattr(gateway.jvm.io.citrine.lolo.validation, metric_name)
    if len(args) > 0:
        metric = metric(*args)

    # Convert the data arrays to Java
    y_true_java = send_1D_array(gateway, y_true, True)
    y_pred_java = send_1D_array(gateway, y_pred, True)
    y_std_java = send_1D_array(gateway, y_std, True)

    # Make the prediction result
    pred_result = gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.makeRegressionPredictionResult(y_pred_java, y_std_java)

    # Run the prediction result through the metric
    return metric.evaluate(pred_result, y_true_java, rng)


def root_mean_squared_error(y_true, y_pred):
    """Compute the root mean squared error
    
    Args:
        y_true ([double]): True value
        y_pred ([double]): Predicted values
    Returns:
        (double): RMSE
    """

    return _call_lolo_merit('RootMeanSquareError', y_true, y_pred)


def standard_confidence(y_true, y_pred, y_std):
    """Fraction of entries that have errors within the predicted confidence interval. 
    
    Args:
        y_true ([double]): True value
        y_pred ([double]): Predicted values
        y_std ([double]): Predicted uncertainty
    Returns:
        (double): standard confidence
    """

    return _call_lolo_merit('StandardConfidence', y_true, y_pred, y_std=y_std)


def standard_error(y_true, y_pred, y_std, rescale=1.0):
    """Root mean square of the error divided by the predicted uncertainty 
    
    Args:
        y_true ([double]): True value
        y_pred ([double]): Predicted values
        y_std ([double]): Predicted uncertainty
        rescale (double): Multiplicative factor with which to rescale error
    Returns:
        (double): standard error
    """

    return _call_lolo_merit('StandardError', y_true, y_pred, None, y_std, float(rescale))


def uncertainty_correlation(y_true, y_pred, y_std, random_seed=None):
    """Measure of the correlation between the predicted uncertainty and error magnitude
    
    Args:
        y_true ([double]): True value
        y_pred ([double]): Predicted values
        y_std ([double]): Predicted uncertainty
        random_seed (int): for reproducibility
    Returns:
        (double):
    """
    return _call_lolo_merit('UncertaintyCorrelation', y_true, y_pred, random_seed=random_seed, y_std=y_std)
