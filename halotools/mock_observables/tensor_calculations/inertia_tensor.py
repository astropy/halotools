"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ('inertia_tensor_per_object', 'inertia_tensors_principal_axes')


def inertia_tensor_per_object(sample1, sample2, weights, smoothing_scale):
    """
    Examples
    --------
    >>> npts1, npts2 = 50, 75
    >>> sample1 = np.random.random((npts1, 3))
    >>> sample2 = np.random.random((npts2, 3))
    >>> weights = np.random.random(npts1)
    >>> smoothing_scale = 0.1
    >>> tensors = inertia_tensor_per_object(sample1, sample2, weights, smoothing_scale)
    """
    raise NotImplementedError()


def _principal_axes_from_matrices(matrices):
    evals, evecs = np.linalg.eigh(matrices)
    return evecs[:, :, 2], evals[:, 2]


def _sphericity_from_matrices(matrices):
    evals, __ = np.linalg.eigh(matrices)
    third_evals = evals[:, 0]
    first_evals = evals[:, 2]
    sphericity = third_evals/first_evals
    return sphericity


def _triaxility_from_matrices(matrices):
    evals, __ = np.linalg.eigh(matrices)
    third_evals = evals[:, 0]
    second_evals = evals[:, 1]
    first_evals = evals[:, 2]
    triaxility = (first_evals**2 - second_evals**2)/(first_evals**2 - third_evals**2)
    return triaxility


def inertia_tensors_principal_axes(sample1, sample2, weights, smoothing_scale):
    """
    """
    return _principal_axes_from_matrices(
            inertia_tensor_per_object(sample1, sample2, weights, smoothing_scale))

