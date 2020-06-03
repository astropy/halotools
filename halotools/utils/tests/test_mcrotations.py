"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..vector_utilities import *
from ..mcrotations import *

__all__ = ('test_random_rotation_3d',
           'test_random_rotation_3d',
           'test_random_perpendicular_directions' )

fixed_seed = 43


def test_random_rotation_3d():
    """
    simple unit test
    """

    npts = 1000
    ndim = 3

    v1 = np.random.random((npts,ndim))
    v2 = random_rotation_3d(v1)

    assert np.shape(v2) == (npts,ndim)
    assert np.all(v1 != v2)


def test_random_rotation_2d():
    """
    simple unit test
    """

    npts = 1000
    ndim = 2

    v1 = np.random.random((npts,ndim))
    v2 = random_rotation_2d(v1)

    assert np.shape(v2) == (npts,ndim)
    assert np.all(v1 != v2)


def test_random_perpendicular_directions():
    """
    check to make sure dot product is zero.
    """

    npts = 1000
    ndim = 3

    v1 = normalized_vectors(np.random.random((npts,ndim)))
    v2 = random_perpendicular_directions(v1)

    assert np.allclose(elementwise_dot(v1,v2),np.zeros(npts))

