"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from rotations.rotate_vector_collection import rotate_vector_collection
from rotations.rotations3d import *

__all__ = ('test_rotation_3d',
           )

fixed_seed = 43

def test_rotation_1():
    """
    single rotation matrix + single set of points
    """
    
    # create a single rotation matrix
    nsets = 1
    ndim = 3
    v1 = np.random.random((nsets,ndim))
    v2 = np.random.random((nsets,ndim))
    rot_m = rotation_matrices_from_vectors(v1,v2)
    rot = rot_m[0]

    # create a single set of vectors
    npts = 1000
    ndim = 3
    v3 = np.random.random((npts,ndim))

    v4 = rotate_vector_collection(rot, v3)

    assert np.shape(v4)==(npts, ndim)


def test_rotation_3():
    """
    nset of rotation matrices + nset of npts of points
    """

    nsets = 2
    ndim = 3
    v1 = np.random.random((nsets,ndim))
    v2 = np.random.random((nsets,ndim))

    rot = rotation_matrices_from_vectors(v1,v2)

    npts = 1000
    ndim = 3
    v3 = np.random.random((nsets, npts, ndim))

    v4 = rotate_vector_collection(rot, v3)
    
    assert np.shape(v4)==(nsets, npts, ndim)
