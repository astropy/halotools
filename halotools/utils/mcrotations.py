"""
A set of rotation utilites that apply monte carlo
roations to collections of 2- and 3-dimensional vectors
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from .vector_utilities import elementwise_norm, elementwise_dot
from .vector_utilities import rotate_vector_collection
from .rotations2d import (
    rotation_matrices_from_angles as rotation_matrices_from_angles_2d,
)
from .rotations3d import (
    rotation_matrices_from_angles as rotation_matrices_from_angles_3d,
)


__all__ = [
    "random_rotation_3d",
    "random_rotation_2d",
    "random_perpendicular_directions",
    "random_unit_vectors_3d",
    "random_unit_vectors_2d",
]
__author__ = ["Duncan Campbell"]


def random_rotation_3d(vectors, seed=None):
    r"""
    Apply a random rotation to a set of 3d vectors.

    Parameters
    ----------
    vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    seed : int, optional
        Random number seed

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    Example
    -------
    Create a random set of 3D unit vectors.

    >>> npts = 1000
    >>> x1 = random_unit_vectors_3d(npts)

    Randomly rotate these vectors.

    >>> x2 = random_rotation_3d(x1)
    """

    with NumpyRNGContext(seed):
        ran_direction = random_unit_vectors_3d(1)[0]
        ran_angle = np.random.random(size=1) * (np.pi)

    ran_rot = rotation_matrices_from_angles_3d(ran_angle, ran_direction)

    return rotate_vector_collection(ran_rot, vectors)


def random_rotation_2d(vectors, seed=None):
    r"""
    Apply a random rotation to a set of 2d vectors.

    Parameters
    ----------
    vectors : ndarray
        Numpy array of shape (npts, 2) storing a collection of 2d vectors

    seed : int, optional
        Random number seed

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, 2) storing a collection of 2d vectors

    Example
    -------
    Create a random set of 2D unit vectors.

    >>> npts = 1000
    >>> x1 = random_unit_vectors_2d(npts)

    Randomly rotate these vectors.

    >>> x2 = random_rotation_2d(x1)
    """

    with NumpyRNGContext(seed):
        ran_angle = np.random.random(size=1) * (np.pi)

    ran_rot = rotation_matrices_from_angles_2d(ran_angle)

    return rotate_vector_collection(ran_rot, vectors)


def random_perpendicular_directions(v, seed=None):
    r"""
    Given an input list of 3D vectors, v, return a list of 3D vectors
    such that each returned vector has unit-length and is
    orthogonal to the corresponding vector in v.

    Parameters
    ----------
    v : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, 3)

    Example
    -------
    Create a random set of 3D unit vectors.

    >>> npts = 1000
    >>> x1 = random_unit_vectors_3d(npts)

    For each vector in x1, create a perpendicular vector

    >>> x2 = random_perpendicular_directions(x1)
    """

    v = np.atleast_2d(v)
    npts = v.shape[0]

    with NumpyRNGContext(seed):
        w = random_unit_vectors_3d(npts)

    vnorms = elementwise_norm(v).reshape((npts, 1))
    wnorms = elementwise_norm(w).reshape((npts, 1))

    e_v = v / vnorms
    e_w = w / wnorms

    v_dot_w = elementwise_dot(e_v, e_w).reshape((npts, 1))

    e_v_perp = e_w - v_dot_w * e_v
    e_v_perp_norm = elementwise_norm(e_v_perp).reshape((npts, 1))

    return e_v_perp / e_v_perp_norm


def random_unit_vectors_3d(npts):
    r"""
    Generate random 3D unit vectors.

    Parameters
    ----------
    npts : int
        number of vectors

    Returns
    -------
    result : numpy.array
        Numpy array of shape (npts, 3) containing random unit vectors
    """

    ndim = 3
    x = np.random.normal(size=(npts, ndim), scale=1.0)
    r = np.sqrt(np.sum((x) ** 2, axis=-1))

    return (1.0 / r[:, np.newaxis]) * x


def random_unit_vectors_2d(npts):
    r"""
    Generate random 2D unit vectors.

    Parameters
    ----------
    npts : int
        number of vectors

    Returns
    -------
    result : numpy.array
        Numpy array of shape (npts, 2) containing random unit vectors
    """

    r = 1.0
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=(npts,))
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return np.vstack((x, y)).T
