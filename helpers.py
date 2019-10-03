"""
This file contains helper functions that I would like to test.
"""
import numpy as np

def checkmat(X):
    """
    Assert that X is a valid matrix.
    """
    assert X.ndim == 2
    assert X.size > 0

def standardize(X):
    """
    Given a data matrix X with samples on the rows and features on the columns,
    center each column and scale each column to unit variance.

    If any of the columns has a standard deviation of 0, a ValueError will be raised.
    """
    checkmat(X)

    n_features = X.shape[1]

    means = np.mean(X, axis=0, keepdims=True)
    assert means.shape == (1, n_features)

    std_devs = np.std(X, axis=0, ddof=0, keepdims=True)
    # Here, I use ddof=0 for no reason other than that sklearn chooses to use ddof=0. See docs:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    assert std_devs.shape == (1, n_features)
    if np.isin(0.0, std_devs):
        raise ValueError("Would divide by zero due to a standard deviation of zero")
    X_standardized = (X - means) / std_devs
    assert X_standardized.shape == X.shape

    assert np.allclose(
        np.mean(X_standardized, axis=0, keepdims=True),
        np.zeros((1, n_features))
    )
    assert np.allclose(
        np.std(X_standardized, axis=0, ddof=0, keepdims=True),
        np.ones((1, n_features))
    )

    return X_standardized
