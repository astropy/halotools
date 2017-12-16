"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext


__all__ = ('elementwise_dot', )


def elementwise_dot(x, y):
    """ Calculate the dot product between
    each pair of elements in two input lists of 3d points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the dot product between each
        pair of corresponding points in x and y.

    Examples
    --------
    >>> npts = int(1e3)
    >>> x = np.random.random((npts, 3))
    >>> y = np.random.random((npts, 3))
    >>> dots = elementwise_dot(x, y)
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.sum(x*y, axis=1)


def elementwise_norm(x):
    """ Calculate the normalization of each element in a list of 3d points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the norm of each 3d point in x.

    Examples
    --------
    >>> npts = int(1e3)
    >>> x = np.random.random((npts, 3))
    >>> norms = elementwise_norm(x)
    """
    x = np.atleast_2d(x)
    return np.sqrt(np.sum(x**2, axis=1))
