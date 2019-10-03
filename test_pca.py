"""
This file contains sanity-checks to verify my assumptions about PCA.
Run the tests using pytest.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helpers import standardize

def test_main(X):
    assert X.ndim == 2
    assert X.size > 0

    n_samples = X.shape[0]
    n_features = X.shape[1]

    X = standardize(X)

def test_standardize(X):
    scaler = StandardScaler()