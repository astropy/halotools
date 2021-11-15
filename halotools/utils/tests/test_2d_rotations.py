"""
"""
import numpy as np
from ..vector_utilities import rotate_vector_collection
from ..rotations2d import rotation_matrices_from_vectors
from ..rotations2d import rotation_matrices_from_angles
from ..rotations2d import rotation_matrices_from_basis

__all__ = (
    "test_rotation_matrices_from_vectors_1",
    "test_rotation_matrices_from_vectors_2",
    "test_rotation_matrices_from_angles",
    "test_rotation_matrices_from_basis_1",
    "test_rotation_matrices_from_basis_2",
)

fixed_seed = 43


def test_rotation_matrices_from_vectors_1():
    """
    test to make sure null rotations return identiy matrix
    """

    N = 1000

    v1 = np.random.random((N, 2))
    rot_m = rotation_matrices_from_vectors(v1, v1)
    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_vectors_2():
    """
    validate 90 degree rotation result
    """

    N = 1000

    v1 = np.zeros((N, 2))
    v1[:, 0] = 1
    v2 = np.zeros((N, 2))
    v2[:, 1] = 1

    rot_m = rotation_matrices_from_vectors(v1, v2)

    v3 = rotate_vector_collection(rot_m, v1)

    assert np.allclose(v2, v3)


def test_rotation_matrices_from_angles():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 2

    rot_m = rotation_matrices_from_angles(np.zeros(npts))

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_basis_1():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 2

    ux = np.zeros((npts, ndim))
    ux[:, 0] = 1.0
    uy = np.zeros((npts, ndim))
    uy[:, 1] = 1.0

    rot_m = rotation_matrices_from_basis(ux, uy)

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_basis_2():
    """
    validate 90 degree rotation result
    """

    npts = 1000
    ndim = 2

    ux = np.zeros((npts, ndim))
    ux[:, 1] = 1.0  # carteisian +y-axis
    uy = np.zeros((npts, ndim))
    uy[:, 0] = -1.0  # carteisian -x-axis

    rot_m = rotation_matrices_from_basis(ux, uy)

    # rotate (0,1) by 45 degrees counter-clockwise.
    v = rotate_vector_collection(rot_m, ux)

    assert np.allclose(v, uy)
