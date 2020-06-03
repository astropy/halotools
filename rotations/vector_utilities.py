r"""
A set of vector calculations to aid in rotation calculations
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np

__all__=['elementwise_dot', 'elementwise_norm', 'normalized_vectors',
         'angles_between_list_of_vectors']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


def normalized_vectors(vectors):
    r"""
    Return a unit-vector for each n-dimensional vector in the input list of n-dimensional points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional points

    Returns
    -------
    normed_x : ndarray
        Numpy array of shape (npts, ndim)

    Examples
    --------
    >>> npts = int(1e3)
    >>> ndim = 3
    >>> x = np.random.random((npts, ndim))
    >>> normed_x = normalized_vectors(x)
    """

    vectors = np.atleast_2d(vectors)
    npts = vectors.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        return vectors/elementwise_norm(vectors).reshape((npts, -1))


def elementwise_norm(x):
    r"""
    Calculate the normalization of each element in a list of n-dimensional points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the norm of each n-dimensional point in x.

    Examples
    --------
    >>> npts = int(1e3)
    >>> ndim = 3
    >>> x = np.random.random((npts, ndim))
    >>> norms = elementwise_norm(x)
    """

    x = np.atleast_2d(x)
    return np.sqrt(np.sum(x**2, axis=1))


def elementwise_dot(x, y):
    r"""
    Calculate the dot product between
    each pair of elements in two input lists of n-dimensional points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of dimensional points

    y : ndarray
        Numpy array of shape (npts, ndim) storing a collection of dimensional points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the dot product between each
        pair of corresponding points in x and y.

    Examples
    --------
    >>> npts = int(1e3)
    >>> ndim = 3
    >>> x = np.random.random((npts, ndim))
    >>> y = np.random.random((npts, ndim))
    >>> dots = elementwise_dot(x, y)
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.sum(x*y, axis=1)


def angles_between_list_of_vectors(v0, v1, tol=1e-3, vn=None):
    r""" Calculate the angle between a collection of n-dimensional vectors

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-D vectors
        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-D vectors
        Note that the normalization of `v1` will be ignored.

    tol : float, optional
        Acceptable numerical error for errors in angle.
        This variable is only used to round off numerical noise that otherwise
        causes exceptions to be raised by the inverse cosine function.
        Default is 0.001.

    n1 : ndarray
        normal vector

    Returns
    -------
    angles : ndarray
        Numpy array of shape (npts, ) storing the angles between each pair of
        corresponding points in v0 and v1.

        Returned values are in units of radians spanning [0, pi].

    Examples
    --------
    >>> npts = int(1e4)
    >>> ndim = 3
    >>> v0 = np.random.random((npts, ndim))
    >>> v1 = np.random.random((npts, ndim))
    >>> angles = angles_between_list_of_vectors(v0, v1)
    """

    dot = elementwise_dot(normalized_vectors(v0), normalized_vectors(v1))
    
    if vn is None:
        #  Protect against tiny numerical excesses beyond the range [-1 ,1]
        mask1 = (dot > 1) & (dot < 1 + tol)
        dot = np.where(mask1, 1., dot)
        mask2 = (dot < -1) & (dot > -1 - tol)
        dot = np.where(mask2, -1., dot)
        a = np.arccos(dot)
    else:
        cross = np.cross(v0,v1)
        a = np.arctan2(elementwise_dot(cross, vn), dot)
    
    return a   


