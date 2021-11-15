r"""
A set of vector rotation utilites for manipulating 2-dimensional vectors
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from .vector_utilities import (
    elementwise_dot,
    elementwise_norm,
    normalized_vectors,
    angles_between_list_of_vectors,
)


__all__ = [
    "rotation_matrices_from_angles",
    "rotation_matrices_from_vectors",
    "rotation_matrices_from_basis",
]
__author__ = ["Duncan Campbell", "Andrew Hearin"]


def rotation_matrices_from_angles(angles):
    r"""
    Calculate a collection of rotation matrices defined by
    an input collection of rotation angles and rotation axes.

    Parameters
    ----------
    angles : ndarray
        Numpy array of shape (npts, ) storing a collection of rotation angles

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 2, 2) storing a collection of rotation matrices

    Examples
    --------
    >>> from halotools.utils.mcrotations import random_unit_vectors_2d
    >>> npts = int(1e4)
    >>> angles = np.random.uniform(-np.pi/2., np.pi/2., npts)
    >>> rotation_matrices = rotation_matrices_from_angles(angles)

    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 2d vectors
    """

    angles = np.atleast_1d(angles)
    npts = len(angles)

    sina = np.sin(angles)
    cosa = np.cos(angles)

    R = np.zeros((npts, 2, 2))
    R[:, 0, 0] = cosa
    R[:, 1, 1] = cosa

    R[:, 0, 1] = -sina
    R[:, 1, 0] = sina

    return R


def rotation_matrices_from_vectors(v0, v1):
    r"""
    Calculate a collection of rotation matrices defined by the unique
    transformation rotating v1 into v2.

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, 2) storing a collection of initial vector orientations.

        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, 2) storing a collection of final vectors.

        Note that the normalization of `v1` will be ignored.

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 2, 2) rotating each v0 into the corresponding v1

    Examples
    --------
    >>> npts = int(1e4)
    >>> v0 = np.random.random((npts, 2))
    >>> v1 = np.random.random((npts, 2))
    >>> rotation_matrices = rotation_matrices_from_vectors(v0, v1)

    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 2d vectors

    """
    v0 = normalized_vectors(v0)
    v1 = normalized_vectors(v1)

    # use the atan2 function to get the direction of rotation right
    angles = np.arctan2(v0[:, 0], v0[:, 1]) - np.arctan2(v1[:, 0], v1[:, 1])

    return rotation_matrices_from_angles(angles)


def rotation_matrices_from_basis(ux, uy, tol=np.pi / 1000.0):
    r"""
    Calculate a collection of rotation matrices defined by an input collection
    of basis vectors.

    Parameters
    ----------
    ux : array_like
        Numpy array of shape (npts, 2) storing a collection of unit vectors

    uy : array_like
        Numpy array of shape (npts, 2) storing a collection of unit vectors

    tol : float, optional
        angular tolerance for orthogonality of the input basis vectors in radians.

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 2, 2) storing a collection of rotation matrices

    Example
    -------
    Let's build a rotation matrix that roates from a frame
    rotated by 45 degrees to the standard frame.

    >>> u1 = [np.sqrt(2), np.sqrt(2)]
    >>> u2 = [np.sqrt(2), -1.0*np.sqrt(2)]
    >>> rot = rotation_matrices_from_basis(u1, u2)

    Notes
    -----
    The rotation matrices transform from the Cartesian frame defined by the standard
    basis vectors,

    .. math::
        \u_1=(1,0)
        \u_2=(0,1)

    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 2d vectors
    """

    N = np.shape(ux)[0]

    # assume initial unit vectors are the standard ones
    ex = np.array([1.0, 0.0] * N).reshape(N, 2)
    ey = np.array([0.0, 1.0] * N).reshape(N, 2)

    ux = normalized_vectors(ux)
    uy = normalized_vectors(uy)

    d_theta = angles_between_list_of_vectors(ux, uy)
    if np.any((np.pi / 2.0 - d_theta) > tol):
        msg = "At least one set of basis vectors are not orthoginal to within the specified tolerance."
        raise ValueError(msg)

    r_11 = elementwise_dot(ex, ux)
    r_12 = elementwise_dot(ex, uy)

    r_21 = elementwise_dot(ey, ux)
    r_22 = elementwise_dot(ey, uy)

    r = np.zeros((N, 2, 2))
    r[:, 0, 0] = r_11
    r[:, 0, 1] = r_12
    r[:, 1, 0] = r_21
    r[:, 1, 1] = r_22

    return r
