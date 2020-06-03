"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from rotations.rotate_vector_collection import rotate_vector_collection
from rotations.rotations3d import *

__all__ = ('test_rotation_matrices_from_vectors',
           'test_rotation_matrices_from_angles',
           'test_rotation_matrices_from_basis' )

fixed_seed = 43


def test_rotation_matrices_from_vectors():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 3
    v1 = np.random.random((npts,ndim))

    rot_m = rotation_matrices_from_vectors(v1,v1)

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_angles():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 3

    uz = np.zeros((npts,ndim))
    uz[:,2] = 1.0

    rot_m = rotation_matrices_from_angles(np.zeros(npts), uz)

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_basis():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 3

    ux = np.zeros((npts,ndim))
    ux[:,0] = 1.0
    uy = np.zeros((npts,ndim))
    uy[:,1] = 1.0
    uz = np.zeros((npts,ndim))
    uz[:,2] = 1.0

    rot_m = rotation_matrices_from_basis(ux, uy, uz)

    assert np.all(~np.isnan(rot_m))
