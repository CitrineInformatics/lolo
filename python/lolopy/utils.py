import sys

import numpy as np


def send_feature_array(gateway, X):
    """Send a feature array to the JVM
    
    Args:
        gateway (JavaGateway): Connection the JVM
        X ([[double]]): 2D array of features
    Returns:
        (Seq[Vector[Double]]) Reference to feature array in JVM
    """
    # Convert X to a numpy array
    X = np.array(X, dtype=np.float64)
    big_end = sys.byteorder == "big"

    # Send X to the JVM
    X_java = gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.getFeatureArray(X.tobytes(), X.shape[1],
                                                                                    big_end)
    return X_java


def send_1D_array(gateway, y, is_float):
    """Send a feature array to the JVM
    
    Args:
        gateway (JavaGateway): Connection the JVM
        y ([[double]]): 1D array to be sent
        is_float (bool): Whether to send data as a float
    Returns:
        (Seq[Vector[Double]]) Reference to feature array in JVM
    """
    # Convert X to a numpy array
    y = np.array(y, dtype=np.float64 if is_float else np.int32)
    big_end = sys.byteorder == "big"

    # Send X to the JVM
    y_java = gateway.jvm.io.citrine.lolo.util.LoloPyDataLoader.get1DArray(y.tobytes(), is_float, big_end)
    return y_java