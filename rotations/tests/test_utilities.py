"""
test suite for vector_utilities.py
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from rotations.vector_utilities import *
from rotations.rotate_vector_collection import rotate_vector_collection
from rotations.rotations2d import rotation_matrices_from_angles as rotation_matrices_from_angles_2d
from rotations.rotations3d import rotation_matrices_from_angles as rotation_matrices_from_angles_3d
from rotations.rotations3d import vectors_normal_to_planes

__all__ = ('test_normalized_vectors', 'test_elementwise_norm', 'test_elementwise_dot',
           'test_angles_between_list_of_vectors')

fixed_seed = 43


def test_normalized_vectors():
    """
    test that normalized square vectors sum to 1
    """

    # 2D vectors
    npts = 100
    ndim = 2
    v1 = np.random.random((npts,ndim))

    result = normalized_vectors(v1)

    assert np.allclose(np.sum(result**2, axis=-1), np.ones(npts))

    # 3D vectors
    npts = 100
    ndim = 3
    v1 = np.random.random((npts,ndim))

    result = normalized_vectors(v1)

    assert np.allclose(np.sum(result**2, axis=-1), np.ones(npts))


def test_elementwise_norm():
    """
    test that normalized vectors norm are 1
    """

    # 2D vectors
    npts = 100
    ndim = 2
    v1 = normalized_vectors(np.random.random((npts,ndim)))

    result = elementwise_norm(v1)

    assert np.allclose(result, np.ones(npts))

    # 3D vectors
    npts = 100
    ndim = 3
    v1 = normalized_vectors(np.random.random((npts,ndim)))

    result = elementwise_norm(v1)

    assert np.allclose(result, np.ones(npts))


def test_elementwise_dot():
    """
    test that perpendicular vectors' dot product is 0.0
    test that parallel vectors' dot product are 1.0
    """

    # 2D vectors
    npts = 100
    ndim = 2
    v1 = normalized_vectors(np.random.random((npts,ndim)))

    # get a set of perpendicular vectors
    rot = rotation_matrices_from_angles_2d(np.ones(npts)*np.pi/2.0)
    v2 = rotate_vector_collection(rot, v1)

    # assert the dot products are all zero
    assert np.allclose(elementwise_dot(v1,v2),np.zeros(npts))
    # assert the dot products are all one
    assert np.allclose(elementwise_dot(v1,v1),np.ones(npts))


    # 3D vectors
    npts = 100
    ndim = 3
    v1 = normalized_vectors(np.random.random((npts,ndim)))
    v2 = np.random.random((npts,ndim))
    v3 = vectors_normal_to_planes(v1, v2)

    # get a set of vectors rotated by 90 degrees
    rot = rotation_matrices_from_angles_3d(np.ones(npts)*np.pi/2.0, v3)
    v4 = rotate_vector_collection(rot, v1)

    # assert the dot products are all zero
    assert np.allclose(elementwise_dot(v1,v4),np.zeros(npts))
    # assert the dot products are all one
    assert np.allclose(elementwise_dot(v1,v1),np.ones(npts))


def test_angles_between_list_of_vectors():
    """
    rotate a vector collection by a random angle and make sure the resulting angles
    between the stes of vectors are consistent
    """

    # 2D vectors
    npts = 100
    ndim = 2
    v1 = np.random.random((npts,ndim))

    # get a set of vectors rotated by random angles
    angles = np.random.uniform(-np.pi/2.0, np.pi/2.0, npts)
    rot = rotation_matrices_from_angles_2d(angles)
    v2 = rotate_vector_collection(rot, v1)

    # assert the dot products are all zero
    assert np.allclose(angles_between_list_of_vectors(v1,v2),np.fabs(angles))


    # 3D vectors
    npts = 100
    ndim = 3
    v1 = np.random.random((npts,ndim))
    v2 = np.random.random((npts,ndim))
    v3 = vectors_normal_to_planes(v1, v2)

    # get a set of vectors rotated by random angles
    angles = np.random.uniform(-np.pi/2.0, np.pi/2.0, npts)
    rot = rotation_matrices_from_angles_3d(angles, v3)
    v4 = rotate_vector_collection(rot, v1)

    # assert the dot products are all zero
    assert np.allclose(angles_between_list_of_vectors(v1,v4),np.fabs(angles))




