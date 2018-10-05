"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..rotations2d import *

__all__ = ('test_rotation_matrices_from_vectors', )

fixed_seed = 43


def test_rotation_matrices_from_vectors_1():
    """
    test to make sure null rotations return identiy matrix
    """

    N = 1000
    
    v1 = np.random.random((N,2))
    rot_m = rotation_matrices_from_vectors(v1,v1)
    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_vectors():
    """
    validate 90 degree rotation result
    """

    N = 1000

    v1 = np.zeros((N,2))
    v1[:,0] = 1
    v2 = np.zeros((N,2))
    v2[:,1] = 1

    rot_m = rotation_matrices_from_vectors(v1,v2)

    v3 = rotate_vector_collection(rot_m, v1)

    assert np.allclose(v2,v3)