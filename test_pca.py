"""
This file contains sanity-checks to verify my assumptions about PCA.
Run the tests using pytest.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pytest

from helpers import (standardize, checkmat)

Xlist = [
    np.array([[1, 2, 3],
              [2, 3, 4]]),
    np.array([[1, 2],
              [2, 1]])
]

@pytest.mark.parametrize(
    "X,newdim", 
    list(zip(Xlist, [1, 1]))
)
def test_main(X, newdim):
    checkmat(X)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    assert newdim <= n_features

    X_scaled = standardize(X)

    # Method 1 of doing PCA: compute eigenpairs of the covariance matrix (X^T times X)
    XTX = X_scaled.T @ X_scaled
    assert XTX.shape == (n_features, n_features)
    (eigvals, eigvecs) = np.linalg.eig(XTX)
    assert eigvals.shape == (n_features,)
    assert eigvecs.shape == (n_features, n_features) # the eigvecs are on the columns
    # The learned model consists of the eigvecs corresponding to the newdim-highest eigvals. 
    # This requires sorting by the eigenvals.
    indices = np.argsort(eigvals)
    assert indices.shape == (n_features,)
    top_indices = indices[-newdim:]
    assert top_indices.shape == (newdim,)
    top_eigvecs = eigvecs[:, top_indices]
    assert top_eigvecs.shape == (n_features, newdim)



@pytest.mark.parametrize("X", Xlist)
def test_standardize(X):
    checkmat(X)

    X_mine = standardize(X)

    scaler = StandardScaler()
    X_sklearn = scaler.fit_transform(X)
    
    assert np.allclose(X_mine, X_sklearn)

def test_sort():
    """
    Tests that argsort + numpy advanced indexing work like I think they do.
    This just makes sure that I'm getting the right eigenvectors.
    """
    eigvals = np.array([1, 4, 2, 0, 3])
    eigvecs = np.array([[1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10]])
    assert eigvecs.shape == (2, 5)

    indices = np.argsort(eigvals)
    assert np.array_equal(
        indices,
        np.array([3, 0, 2, 4, 1])
    )
    assert np.array_equal(
        eigvals[indices],
        np.array([0, 1, 2, 3, 4])
    )

    top3 = indices[-3:]
    top_eigvecs = eigvecs[:, top3]
    assert top_eigvecs.shape == (2, 3)
    assert np.array_equal(
        top_eigvecs,
        np.array([[3, 5, 2],
                  [8, 10, 7]])
    )
