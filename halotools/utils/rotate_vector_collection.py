"""
A function to rotate collections of n-dimensional vectors
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
    Given a collection of rotation matrices and a collection of n-dimensional vectors,
    apply an asscoiated matrix to rotate corresponding vector(s).

    Parameters
    ----------
    rotation_matrices : ndarray
        The options are:
        1.) array of shape (npts, ndim, ndim) storing a collection of rotation matrices.
        2.) array of shape (ndim, ndim) storing a single rotation matrix
        
    vectors : ndarray
        The corresponding options for above are:
        1.) array of shape (npts, ndim) storing a collection of ndim-dimensional vectors
        2.) array of shape (npts, ndim) storing a collection of ndim-dimensional vectors

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-dimensional vectors

    Notes
    -----
    This function is set up to preform either rotation operations on a single collection \
    of vectors, either applying a single rotation matrix to all vectors in the collection,
    or applying a unique rotation matrix to each vector in the set.

    The behavior of the function is determined by the arguments supplied by the user.

    Examples
    --------
    In this example, we'll randomly generate two sets of unit-vectors, `v0` and `v1`.
    We'll use the `rotation_matrices_from_vectors` function to generate the
    rotation matrices that rotate each `v0` into the corresponding `v1`.
    Then we'll use the `rotate_vector_collection` function to apply each
    rotation, and verify that we recover each of the `v1`.

    >>> from halotools.utils.rotations3d import rotation_matrices_from_vectors
    >>> from halotools.utils import normalized_vectors
    >>> npts, ndim = int(1e4), 3
    >>> v0 = normalized_vectors(np.random.random((npts, ndim)))
    >>> v1 = normalized_vectors(np.random.random((npts, ndim)))
    >>> rotation_matrices = rotation_matrices_from_vectors(v0, v1)
    >>> v2 = rotate_vector_collection(rotation_matrices, v0)
    >>> assert np.allclose(v1, v2)
    """

    ndim_rotm = np.shape(rotation_matrices)[-1]
    ndim_vec = np.shape(vectors)[-1]
    assert ndim_rotm==ndim_vec

    if len(np.shape(vectors))==2:
        ntps, ndim = np.shape(vectors)
        nsets = 0
    elif len(np.shape(vectors))==3:
        nsets, ntps, ndim = np.shape(vectors)

    # apply same rotation matrix to all vectors
    if (len(np.shape(rotation_matrices)) == 2):
        if nsets == 1:
            vectors = vectors[0]
        return np.dot(rotation_matrices, vectors.T).T
    # rotate each vector by associated rotation matrix
    else:
        ein_string = 'ijk,ik->ij'
        n1, ndim = np.shape(vectors)

        try:
            return np.einsum(ein_string, rotation_matrices, vectors, optimize=optimize)
        except TypeError:
            return np.einsum(ein_string, rotation_matrices, vectors)


