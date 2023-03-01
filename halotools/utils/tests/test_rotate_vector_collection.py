"""
"""
import numpy as np
from ..vector_utilities import rotate_vector_collection
from ..rotations3d import rotation_matrices_from_vectors

__all__ = ("test_rotation_1", "test_rotation_2", "test_rotation_3")

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

def test_rotation_3():
    """
    test option 3: rotate n1 sets of axes, each set containing n2 axes of ndim=3 dimensions.
    This uses n1 rotation matrices, each of which will rotate all axes in the corresponding set.
    e.g. giving three rotation matrices and three sets of axes, each set containing four 3D axes, the first matrix will rotate all four axes in the first set, and so on
    This test is for git commit 4f8ed3ab1e24bd6943fc788a1ae826189ba5921e
    """
    
    # make 3 rotation matrices
    nsets = 3
    ndim = 3
    v1 = np.random.random((nsets, ndim))
    v2 = np.random.random((nsets, ndim))
    rot = rotation_matrices_from_vectors(v1, v2)
    
    # create three sets of vectors
    npts = 1000
    ndim = 3
    v3 = [ np.random.random((npts, ndim)), np.random.random((npts, ndim)), np.random.random((npts, ndim)) ]
    
    v4 = rotate_vector_collection(rot, v3)
    assert( np.shape(v4) == (nsets, npts, ndim) )
    
    # Test that each set is equivalent to having done it individually using a single rotation matrix
    for i in range(len(rot)):
        v5 = rotate_vector_collection([rot[i]], v3[i])
        assert( ( v4[i] == v5 ).all() )