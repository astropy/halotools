"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..matrix_operations_3d import *

__all__ = ('test_rotation_matrices_from_vectors', )

fixed_seed = 43


def test_rotation_matrices_from_vectors():
    """
    test to make sure null rotations return identiy matrix
    """

    N = 1000
    v1 = np.random.random((N,3))

    rot_m = rotation_matrices_from_vectors(v1,v1)
    
    print(np.all(~np.isnan(rot_m)))
    assert np.all(~np.isnan(rot_m))


