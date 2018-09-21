"""
A set of vector calculations to aid in rotation calculations
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from rotations import *


__all__=['elementwise_dot', 'elementwise_norm', 'normalized_vectors',
         'angles_between_list_of_vectors', 'vectors_normal_to_planes',
         'vectors_between_list_of_vectors', 'project_onto_plane']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


def vectors_between_list_of_vectors(x, y, p):
    r"""
    Starting from two input lists of vectors, return a list of unit-vectors
    that lie in the same plane as the corresponding input vectors,
    and where the input `p` controls the angle between
    the returned vs. input vectors.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `x` will be ignored.

    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `y` will be ignored.

    p : ndarray
        Numpy array of shape (npts, ) storing values in the closed interval [0, 1].
        For values of `p` equal to zero, the returned vectors will be
        exactly aligned with the input `x`; when `p` equals unity, the returned
        vectors will be aligned with `y`.

    Returns
    -------
    v : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d unit-vectors
        lying in the plane spanned by `x` and `y`. The angle between `v` and `x`
        will be equal to :math:`p*\theta_{\rm xy}`.

    Examples
    --------
    >>> npts = int(1e4)
    >>> x = np.random.random((npts, 3))
    >>> y = np.random.random((npts, 3))
    >>> p = np.random.uniform(0, 1, npts)
    >>> v = vectors_between_list_of_vectors(x, y, p)
    >>> angles_xy = angles_between_list_of_vectors(x, y)
    >>> angles_xp = angles_between_list_of_vectors(x, v)
    >>> assert np.allclose(angles_xy*p, angles_xp)
    """
    assert np.all(p >= 0), "All values of p must be non-negative"
    assert np.all(p <= 1), "No value of p can exceed unity"

    z = vectors_normal_to_planes(x, y)
    theta = angles_between_list_of_vectors(x, y)
    angles = p*theta
    rotation_matrices = rotation_matrices_from_angles(angles, z)
    return normalized_vectors(rotate_vector_collection(rotation_matrices, x))


def vectors_normal_to_planes(x, y):
    r""" Given a collection of 3d vectors x and y,
    return a collection of 3d unit-vectors that are orthogonal to x and y.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `x` will be ignored.

    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `y` will be ignored.

    Returns
    -------
    z : ndarray
        Numpy array of shape (npts, 3). Each 3d vector in z will be orthogonal
        to the corresponding vector in x and y.

    Examples
    --------
    >>> npts = int(1e4)
    >>> x = np.random.random((npts, 3))
    >>> y = np.random.random((npts, 3))
    >>> normed_z = angles_between_list_of_vectors(x, y)

    """
    return normalized_vectors(np.cross(x, y))


def angles_between_list_of_vectors(v0, v1, tol=1e-3):
    r""" Calculate the angle between a collection of 3d vectors

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `v1` will be ignored.

    tol : float, optional
        Acceptable numerical error for errors in angle.
        This variable is only used to round off numerical noise that otherwise
        causes exceptions to be raised by the inverse cosine function.
        Default is 0.001.

    Returns
    -------
    angles : ndarray
        Numpy array of shape (npts, ) storing the angles between each pair of
        corresponding points in v0 and v1.

        Returned values are in units of radians spanning [0, pi].

    Examples
    --------
    >>> npts = int(1e4)
    >>> v0 = np.random.random((npts, 3))
    >>> v1 = np.random.random((npts, 3))
    >>> angles = angles_between_list_of_vectors(v0, v1)
    """
    dot = elementwise_dot(normalized_vectors(v0), normalized_vectors(v1))

    #  Protect against tiny numerical excesses beyond the range [-1 ,1]
    mask1 = (dot > 1) & (dot < 1 + tol)
    dot = np.where(mask1, 1., dot)
    mask2 = (dot < -1) & (dot > -1 - tol)
    dot = np.where(mask2, -1., dot)

    return np.arccos(dot)


def normalized_vectors(vectors):
    r""" Return a unit-vector for each 3d vector in the input list of 3d points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    normed_x : ndarray
        Numpy array of shape (npts, 3)

    Examples
    --------
    >>> npts = int(1e3)
    >>> x = np.random.random((npts, 3))
    >>> normed_x = normalized_vectors(x)
    """
    vectors = np.atleast_2d(vectors)
    npts = vectors.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        return vectors/elementwise_norm(vectors).reshape((npts, -1))


def elementwise_norm(x):
    r""" Calculate the normalization of each element in a list of 3d points.

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


def elementwise_dot(x, y):
    r""" Calculate the dot product between
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


def project_onto_plane(x1, x2):
    r"""
    Given a collection of 3D vectors, x1 and x2, project each vector
    in x1 onto the plane normal to the corresponding vector x2

    Parameters
    ----------
    x1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    x2 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    """

    n = normalized_vectors(x2)
    d = elementwise_dot(x1,n)
    
    return x - d[:,np.newaxis]*n

