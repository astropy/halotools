"""
"""
import numpy as np
from ..vector_utilities import rotate_vector_collection
from ..rotations3d import rotation_matrices_from_vectors

__all__ = ("test_rotation_1",)

fixed_seed = 43


def test_rotation_1():
    """
    test option 1: single rotation matrix + set of points
    """

    # create a single rotation matrix
    nsets = 1
    ndim = 3
    v1 = np.random.random((nsets, ndim))
    v2 = np.random.random((nsets, ndim))
    rot_m = rotation_matrices_from_vectors(v1, v2)
    rot = rot_m[0]

    # create a set of vectors
    npts = 1000
    ndim = 3
    v3 = np.random.random((npts, ndim))

    v4 = rotate_vector_collection(rot, v3)

    assert np.shape(v4) == (npts, ndim)


def test_rotation_2():
    """
    test option 2: n rotation matrices + n points
    """

    npts = 1000
    ndim = 3
    v1 = np.random.random((npts, ndim))
    v2 = np.random.random((npts, ndim))

    rot = rotation_matrices_from_vectors(v1, v2)

    v3 = np.random.random((npts, ndim))

    v4 = rotate_vector_collection(rot, v3)

    assert np.shape(v4) == (npts, ndim)
