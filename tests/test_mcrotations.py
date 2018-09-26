"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..mcrotations import *

__all__ = ('test_random_rotation_3d',
           'test_random_rotation_3d',
           'test_random_perpendicular_directions' )

fixed_seed = 43


def test_random_rotation_3d():
    """
    """

    npts = 1000
    ndim = 3

    v1 = np.random.random((npts,ndim))
    v2 = random_rotation_3d(v1)

    assert np.all(v1 != v2)


def test_random_rotation_2d():
    """
    """

    npts = 1000
    ndim = 2

    v1 = np.random.random((npts,ndim))
    v2 = random_rotation_2d(v1)

    assert np.all(v1 != v2)
