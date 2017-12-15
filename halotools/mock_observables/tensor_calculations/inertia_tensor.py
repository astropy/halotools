"""
"""
import numpy as np



__all__ = ('inertia_tensor_per_object', )


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
    pass
