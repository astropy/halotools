"""
A function to rotate collectios of n-dimensional vectors
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from .vector_utilities import (elementwise_dot, elementwise_norm,
                               normalized_vectors, angles_between_list_of_vectors)


__all__=['rotate_vector_collection',]
__author__ = ['Duncan Campbell', 'Andrew Hearin']

def rotate_vector_collection(rotation_matrices, vectors, optimize=False):
    r"""
    Given a collection of rotation matrices and a collection of 3d vectors,
    apply each an asscoiated matrix to rotate a corresponding vector.

    Parameters
    ----------
    rotation_matrices : ndarray
        The options are:
        1.) array of shape (npts, ndim, ndim) storing a collection of rotation matrices.
        2.) array of shape (ndim, ndim) storing a single rotation matrix
        3.) array of shape (nsets, ndim, ndim) storing a collection of rotation matrices.

    vectors : ndarray
        The corresponding options for
        1.) array of shape (npts, ndim) storing a collection of 3d vectors
        2.) array of shape (npts, ndim) storing a collection of 3d vectors
        3.) array of shape (nsets, npts, ndim) storing a collection of 3d vectors

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

    ndim = np.shape(rotation_matrices)[-1]

    # apply same rotation matrix to all vectors
    if len(np.shape(rotation_matrices)) == 2:
        return np.dot(rotation_matrices, vectors.T).T
    # rotate each vector by associated rotation matrix
    else:
        # n1 sets of n2 vectors of ndim dimension
        if len(np.shape(vectors))==3:
            ein_string = 'ikl,ijl->ijk'
            n1, n2, ndim = np.shape(vectors)
        # n1 vectors of ndim dimension
        elif len(np.shape(vectors))==2:
            ein_string = 'ijk,ik->ij'
            n1, ndim = np.shape(vectors)
        
        assert np.shape(rotation_matrices)==(n1,ndim,ndim)
        
        try:
            return np.einsum(ein_string, rotation_matrices, vectors, optimize=optimize)
        except TypeError:
            return np.einsum(ein_string, rotation_matrices, vectors)


