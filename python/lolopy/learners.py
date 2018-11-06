from lolopy.loloserver import get_java_gateway
from sklearn.base import BaseEstimator, RegressorMixin
from py4j.java_collections import ListConverter
import numpy as np


class RandomForest(BaseEstimator, RegressorMixin):

    def __init__(self, num_trees=-1, useJackknife=True, subsetStrategy=4):
        """Initialize the RandomForest

        Args:
            num_trees (int): Number of trees to use in the forest
        """
        super(BaseEstimator, self).__init__()

        # Get JVM for this object
        self.gateway = get_java_gateway()

        # Store the variables
        self.num_trees = num_trees
        self.useJackknife = useJackknife
        self.subsetStrategy = subsetStrategy

        # Create a placeholder for the model
        self.model_ = None

    def fit(self, X, y, weights=None):

        # Instantiate the JVM object
        #  TODO: Figure our a more succinct way of dealing with optional arguments/Option values
        learner = self.gateway.jvm.io.citrine.lolo.learners.RandomForest(
            self.num_trees, self.useJackknife,
            getattr(self.gateway.jvm.io.citrine.lolo.learners.RandomForest, "$lessinit$greater$default$3")(),
            getattr(self.gateway.jvm.io.citrine.lolo.learners.RandomForest, "$lessinit$greater$default$4")(),
            self.subsetStrategy
        )

        # Convert X and y to Java Objects
        train_data = self.gateway.jvm.java.util.ArrayList(len(y))
        for x_i, y_i in zip(X, y):
            # Copy the X into an array
            x_i_java = ListConverter().convert(x_i, self.gateway._gateway_client)
            x_i_java = self.gateway.jvm.scala.collection.JavaConverters.asScalaBuffer(x_i_java).toVector()
            pair = self.gateway.jvm.scala.Tuple3(x_i_java, float(y_i), 1.0)
            train_data.append(pair)
        train_data = self.gateway.jvm.scala.collection.JavaConverters.asScalaBuffer(train_data)

        # Make the weights
        weights_java = self.gateway.jvm.java.util.ArrayList(len(y))
        if weights is None:
            for i in range(len(y)):
                weights_java.append(1)
        weights_java = self.gateway.jvm.scala.collection.JavaConverters.asScalaBuffer(weights_java)

        # Run the training
        result = learner.train(train_data)

        # Get the model out
        self.model_ = result.getModel()

        return self

    def predict(self, X, return_std=False):

        # Convert X to an array
        X_java = self.gateway.jvm.java.util.ArrayList(len(X))
        for x_i in X:
            x_i_java = ListConverter().convert(x_i, self.gateway._gateway_client)
            x_i_java = self.gateway.jvm.scala.collection.JavaConverters.asScalaBuffer(x_i_java).toVector()
            X_java.append(x_i_java)
        X_java = self.gateway.jvm.scala.collection.JavaConverters.asScalaBuffer(X_java)

        # Get the PredictionResult
        pred_result = self.model_.transform(X_java)

        # Pull out the expected values
        exp_values = pred_result.getExpected()
        y_pred = np.zeros((len(X),))
        for i in range(len(X)):
            y_pred[i] = exp_values.apply(i)

        # If desired, return the uncertainty too
        if return_std:
            # TODO: This part fails on Windows because the NativeSystemBLAS is not found. Fix that
            uncertain = pred_result.getUncertainty().get()
            y_std = np.zeros_like(y_pred)
            for i in range(len(X)):
                y_std[i] = uncertain.apply(i)
            return y_pred, y_std

        # Get the expected values
        return y_pred
