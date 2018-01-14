""" Common functions used to rotate three dimensional vectors
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ('elementwise_dot', 'elementwise_norm', 'normalized_vectors',
        'angles_between_list_of_vectors', 'vectors_normal_to_planes',
        'rotate_vector_collection', 'rotation_matrices_from_angles',
        'rotation_matrices_from_vectors')


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
    return vectors/elementwise_norm(vectors).reshape((npts, -1))


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


def rotation_matrices_from_angles(angles, directions):
    r""" Calculate a collection of rotation matrices defined by
    an input collection of rotation angles and rotation axes.

    Parameters
    ----------
    angles : ndarray
        Numpy array of shape (npts, ) storing a collection of rotation angles

    directions : ndarray
        Numpy array of shape (npts, 3) storing a collection of rotation axes in 3d

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices

    Examples
    --------
    >>> npts = int(1e4)
    >>> angles = np.random.uniform(-np.pi/2., np.pi/2., npts)
    >>> directions = np.random.random((npts, 3))
    >>> rotation_matrices = rotation_matrices_from_angles(angles, directions)

    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 3d vectors

    """
    directions = normalized_vectors(directions)
    angles = np.atleast_1d(angles)
    npts = directions.shape[0]

    sina = np.sin(angles)
    cosa = np.cos(angles)

    R1 = np.zeros((npts, 3, 3))
    R1[:, 0, 0] = cosa
    R1[:, 1, 1] = cosa
    R1[:, 2, 2] = cosa

    R2 = directions[..., None] * directions[:, None, :]
    R2 = R2*np.repeat(1.-cosa, 9).reshape((npts, 3, 3))

    directions *= sina.reshape((npts, 1))
    R3 = np.zeros((npts, 3, 3))
    R3[:, [1, 2, 0], [2, 0, 1]] -= directions
    R3[:, [2, 0, 1], [1, 2, 0]] += directions

    return R1 + R2 + R3


def rotation_matrices_from_vectors(v0, v1):
    r""" Calculate a collection of rotation matrices defined by the unique
    transformation rotating v1 into v2 about the mutually perpendicular axis.

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, 3) storing a collection of initial vector orientations.

        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of final vectors.

        Note that the normalization of `v1` will be ignored.

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) rotating each v0 into the corresponding v1

    Examples
    --------
    >>> npts = int(1e4)
    >>> v0 = np.random.random((npts, 3))
    >>> v1 = np.random.random((npts, 3))
    >>> rotation_matrices = rotation_matrices_from_vectors(v0, v1)

    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 3d vectors

    """
    v0 = normalized_vectors(v0)
    v1 = normalized_vectors(v1)
    directions = vectors_normal_to_planes(v0, v1)
    angles = angles_between_list_of_vectors(v0, v1)

    return rotation_matrices_from_angles(angles, directions)


def rotate_vector_collection(rotation_matrices, vectors, optimize=False):
    r""" Given a collection of rotation matrices and a collection of 3d vectors,
    apply each matrix to rotate the corresponding vector.

    Parameters
    ----------
    rotation_matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices

    vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    Examples
    --------
    In this example, we'll randomly generate two sets of unit-vectors, `v0` and `v1`.
    We'll use the `rotation_matrices_from_vectors` function to generate the
    rotation matrices that rotate each `v0` into the corresponding `v1`.
    Then we'll use the `rotate_vector_collection` function to apply each
    rotation, and verify that we recover each of the `v1`.

    >>> npts = int(1e4)
    >>> v0 = normalized_vectors(np.random.random((npts, 3)))
    >>> v1 = normalized_vectors(np.random.random((npts, 3)))
    >>> rotation_matrices = rotation_matrices_from_vectors(v0, v1)
    >>> v2 = rotate_vector_collection(rotation_matrices, v0)
    >>> assert np.allclose(v1, v2)
    """
    return np.einsum('ijk,ik->ij', rotation_matrices, vectors, optimize=optimize)
