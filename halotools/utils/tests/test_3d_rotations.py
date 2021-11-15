"""
"""
import numpy as np
from ..vector_utilities import angles_between_list_of_vectors
from ..vector_utilities import rotate_vector_collection
from ..rotations3d import rotation_matrices_from_vectors
from ..rotations3d import rotation_matrices_from_angles
from ..rotations3d import rotation_matrices_from_basis
from ..rotations3d import vectors_between_list_of_vectors

__all__ = (
    "test_rotation_matrices_from_vectors",
    "test_rotation_matrices_from_angles",
    "test_rotation_matrices_from_basis_1",
    "test_rotation_matrices_from_basis_2",
    "test_vectors_between_list_of_vectors",
)

fixed_seed = 43


def test_rotation_matrices_from_vectors():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 3
    v1 = np.random.random((npts, ndim))

    rot_m = rotation_matrices_from_vectors(v1, v1)

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_angles():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 3

    uz = np.zeros((npts, ndim))
    uz[:, 2] = 1.0

    rot_m = rotation_matrices_from_angles(np.zeros(npts), uz)

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_basis_1():
    """
    test to make sure null rotations return identiy matrix
    """

    npts = 1000
    ndim = 3

    ux = np.zeros((npts, ndim))
    ux[:, 0] = 1.0
    uy = np.zeros((npts, ndim))
    uy[:, 1] = 1.0
    uz = np.zeros((npts, ndim))
    uz[:, 2] = 1.0

    rot_m = rotation_matrices_from_basis(ux, uy, uz)

    assert np.all(~np.isnan(rot_m))


def test_rotation_matrices_from_basis_2():
    """
    test tolerance feature
    """

    npts = 1000
    ndim = 3

    tol = np.pi / 1000
    epsilon = np.array([tol + 0.01] * npts)

    ux = np.zeros((npts, ndim))
    ux[:, 0] = 1.0
    uy = np.zeros((npts, ndim))
    uy[:, 1] = 1.0
    uz = np.zeros((npts, ndim))
    uz[:, 2] = 1.0

    rot_m = rotation_matrices_from_basis(ux, uy, uz, tol=np.pi / 1000)
    assert np.all(~np.isnan(rot_m))

    # perturbate x
    rot_m = rotation_matrices_from_angles(epsilon, uz)
    ux_prime = rotate_vector_collection(rot_m, ux)

    dtheta = angles_between_list_of_vectors(ux_prime, uy)
    dtheta = np.fabs(np.pi / 2.0 - np.max(dtheta))

    try:
        rot_m = rotation_matrices_from_basis(ux_prime, uy, uz, tol=tol)
        assert (
            True == False
        ), "tolerance set to {0}, but measured misalignment was at least {1}.".format(
            tol, dtheta
        )
    except ValueError:
        pass

    # perturbate y
    rot_m = rotation_matrices_from_angles(epsilon, ux)
    uy_prime = rotate_vector_collection(rot_m, uy)

    dtheta = angles_between_list_of_vectors(ux_prime, uy)
    dtheta = np.fabs(np.pi / 2.0 - np.max(dtheta))

    try:
        rot_m = rotation_matrices_from_basis(ux, uy_prime, uz, tol=tol)
        assert (
            True == False
        ), "tolerance set to {0}, but measured misalignment was at least {1}.".format(
            tol, dtheta
        )
    except ValueError:
        pass

    # perturbate z
    rot_m = rotation_matrices_from_angles(epsilon, uy)
    uz_prime = rotate_vector_collection(rot_m, uz)

    dtheta = angles_between_list_of_vectors(ux_prime, uy)
    dtheta = np.fabs(np.pi / 2.0 - np.max(dtheta))

    try:
        rot_m = rotation_matrices_from_basis(ux, uy, uz_prime, tol=tol)
        assert (
            True == False
        ), "tolerance set to {0}, but measured misalignment was at least {1}.".format(
            tol, dtheta
        )
    except ValueError:
        pass


def test_vectors_between_list_of_vectors():
    """"""

    npts = 1000
    x = np.random.random((npts, 3))
    y = np.random.random((npts, 3))
    p = np.random.uniform(0, 1, npts)

    v = vectors_between_list_of_vectors(x, y, p)

    angles_xy = angles_between_list_of_vectors(x, y)
    angles_xp = angles_between_list_of_vectors(x, v)

    assert np.allclose(angles_xy * p, angles_xp)
