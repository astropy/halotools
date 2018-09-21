"""
A set of rotaiton utilites
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from .vector_calculations import *


__all__=['rotate_vector_collection', 'random_rotation', 'rotation_matrices_from_angles',
         'rotation_matrices_from_vectors']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


def rotate_vector_collection(rotation_matrices, vectors, optimize=False):
    r""" 
    Given a collection of rotation matrices and a collection of 3d vectors,
    apply each matrix to rotate the corresponding vector.
    
    Parameters
    ----------
    rotation_matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices.
        If an array of shape (3, 3) is passed, all the vectors
        are rotated using the same rotation matrix.
    
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

    # apply same rotation matrix to all vectors
    if np.shape(rotation_matrices) == (3, 3):
        return np.dot(rotation_matrices, vectors.T).T
    # rotate each vector by associated rotation matrix
    else:
        try:
            return np.einsum('ijk,ik->ij', rotation_matrices, vectors, optimize=optimize)
        except TypeError:
            return np.einsum('ijk,ik->ij', rotation_matrices, vectors)


def rotation_matrices_from_angles(angles, directions):
    r""" 
    Calculate a collection of rotation matrices defined by
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
    r""" 
    Calculate a collection of rotation matrices defined by the unique
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

    # where angles are 0.0, replace directions with v0
    mask_a = (np.isnan(directions[:,0]) | np.isnan(directions[:,1]) | np.isnan(directions[:,2]))
    mask_b = (angles==0.0)
    mask = mask_a | mask_b
    directions[mask] = v0[mask]

    return rotation_matrices_from_angles(angles, directions)


def rotation3d(ux, uy, uz):
    """
    Calculate a collection of rotation matrices defined by an input collection
    of basis vectors.
    
    Parameters
    ----------
    ux : array_like
        Numpy array of shape (npts, 3) storing a collection of unit vexctors
    
    uy : array_like
        Numpy array of shape (npts, 3) storing a collection of unit vexctors
    
    uz : array_like
        Numpy array of shape (npts, 3) storing a collection of unit vexctors
    
    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices
    """

    N = np.shape(ux)[0]

    # assume initial unit vectors are the standard ones
    ex = np.array([1.0, 0.0, 0.0]*N).reshape(N, 3)
    ey = np.array([0.0, 1.0, 0.0]*N).reshape(N, 3)
    ez = np.array([0.0, 0.0, 1.0]*N).reshape(N, 3)

    ux = normalized_vectors(ux)
    uy = normalized_vectors(uy)
    uz = normalized_vectors(uz)

    r_11 = elementwise_dot(ex, ux)
    r_12 = elementwise_dot(ex, uy)
    r_13 = elementwise_dot(ex, uz)

    r_21 = elementwise_dot(ey, ux)
    r_22 = elementwise_dot(ey, uy)
    r_23 = elementwise_dot(ey, uz)

    r_31 = elementwise_dot(ez, ux)
    r_32 = elementwise_dot(ez, uy)
    r_33 = elementwise_dot(ez, uz)

    r = np.zeros((N, 3, 3))
    r[:,0,0] = r_11
    r[:,0,1] = r_12
    r[:,0,2] = r_13
    r[:,1,0] = r_21
    r[:,1,1] = r_22
    r[:,1,2] = r_23
    r[:,2,0] = r_31
    r[:,2,1] = r_32
    r[:,2,2] = r_33

    return r

