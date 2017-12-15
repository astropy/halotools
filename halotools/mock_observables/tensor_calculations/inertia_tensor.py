"""
"""
import numpy as np



__all__ = ('inertia_tensor_per_object', 'inertia_tensor_principal_axis')


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


def inertia_tensor_principal_axis(sample1, sample2, weights, smoothing_scale):
    """
    """
    inertia_tensors = inertia_tensor_per_object(sample1, sample2, weights, smoothing_scale)
    evals, evecs = np.linalg.eigh(inertia_tensors)
    return evecs[:, :, 2], evals[:, 2]


