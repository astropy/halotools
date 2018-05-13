r"""
"""
import numpy as np
from ..matrix_operations_3d import elementwise_norm
from ..satellite_rotations import rotate_satellite_vectors, calculate_satellite_radial_vector
from ..satellite_rotations import reposition_satellites_from_radial_vectors


def test1():
    r""" Randomly generate vectors and rotations and ensure
    the rotate_satellite_vectors function preserves norm.
    """
    nsats, nhosts = int(1e4), 100
    satellite_hostid = np.random.randint(0, nhosts, nsats)
    satellite_vectors = np.random.uniform(-1, 1, nsats*3).reshape((nsats, 3))
    satellite_rotation_angles = np.random.uniform(-np.pi, np.pi, nsats)
    host_halo_id = np.arange(nhosts)
    host_halo_axis = np.random.uniform(-1, 1, nhosts*3).reshape((nhosts, 3))
    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)
    orig_norms = elementwise_norm(satellite_vectors)
    new_norms = elementwise_norm(new_vectors)
    assert np.allclose(orig_norms, new_norms)


def test2():
    r"""
    All satellite vectors are normalized and point in x-direction.
    All host vectors are normalized and point in z-direction.
    Rotate the matched satellites by (pi, pi/2, -pi/2) and enforce that we get (-x, y, -y)
    """
    satellite_hostid = [1, 3, 0]
    satellite_vectors = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    satellite_rotation_angles = [np.pi, np.pi/2., -np.pi/2.]
    host_halo_id = [0, 1, 2, 3, 4]
    host_halo_axis = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)

    assert new_vectors.shape == (3, 3)
    assert np.allclose(new_vectors[0, :], [-1, 0, 0])
    assert np.allclose(new_vectors[1, :], [0, 1, 0])
    assert np.allclose(new_vectors[2, :], [0, -1, 0])


def test3():
    r"""
    All satellite vectors are normalized and point in y-direction.
    All host vectors are normalized and point in x-direction.
    Rotate the matched satellites by (pi, pi/2, -pi/2) and enforce that we get (-y, z, -z)
    """
    satellite_hostid = [1, 3, 0]
    satellite_vectors = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    satellite_rotation_angles = [np.pi, np.pi/2., -np.pi/2.]
    host_halo_id = [0, 1, 2, 3, 4]
    host_halo_axis = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)

    assert new_vectors.shape == (3, 3)
    assert np.allclose(new_vectors[0, :], [0, -1, 0])
    assert np.allclose(new_vectors[1, :], [0, 0, 1])
    assert np.allclose(new_vectors[2, :], [0, 0, -1])


def test4():
    r""" Concatenate four explicit cases and use nontrivial sequencing of
    satellite_hostid to ensure all four hard-coded examples are explicitly correct
    """
    satellite_hostid = [1, 3, 0, 2]
    satellite_vectors = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
    satellite_rotation_angles = [np.pi, np.pi/2., -np.pi/2., np.pi/2.]
    host_halo_id = [0, 1, 2, 3, 4]
    host_halo_axis = [[1, 0, 0], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [1, 0, 0]]

    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)

    assert new_vectors.shape == (4, 3)
    msg = ("This test4 intends to ensure that the passing test2 and test3\n"
        "have results that propagate through to nontrivial usage of satellite_hostid")
    assert np.allclose(new_vectors[0, :], [-1, 0, 0]), msg
    assert np.allclose(new_vectors[1, :], [0, 0, 1]), msg
    assert np.allclose(new_vectors[2, :], [1, 0, 0]), msg
    assert np.allclose(new_vectors[3, :], [0, 0, -1]), msg


def test_radial_vector1():
    nsats = 100
    nhosts = 10
    Lbox = 1.

    sat_x = np.random.uniform(0, Lbox, nsats)
    sat_y = np.random.uniform(0, Lbox, nsats)
    sat_z = np.random.uniform(0, Lbox, nsats)
    sat_hostid = np.random.randint(0, nhosts, nsats)

    host_halo_x = np.random.uniform(0, Lbox, nhosts)
    host_halo_y = np.random.uniform(0, Lbox, nhosts)
    host_halo_z = np.random.uniform(0, Lbox, nhosts)
    host_halo_id = np.arange(0, nhosts)

    normalized_radial_vectors, radial_distances = calculate_satellite_radial_vector(
            sat_hostid, sat_x, sat_y, sat_z,
            host_halo_id, host_halo_x, host_halo_y, host_halo_z, Lbox)
    norms = elementwise_norm(normalized_radial_vectors)
    assert np.allclose(norms, 1.)


def test_radial_vector2():
    Lbox = 1.

    sat_x = [0.95, 0.95, 0.95]
    sat_y = [0.95, 0.95, 0.95]
    sat_z = [0.95, 0.95, 0.95]
    sat_hostid = [1, 1, 1]

    host_halo_x = [0.9, 0.9, 0.9]
    host_halo_y = [0.9, 0.9, 0.9]
    host_halo_z = [0.9, 0.9, 0.9]
    host_halo_id = [0, 1, 2]

    normalized_radial_vectors, radial_distances = calculate_satellite_radial_vector(
            sat_hostid, sat_x, sat_y, sat_z,
            host_halo_id, host_halo_x, host_halo_y, host_halo_z, Lbox)

    correct_distance = np.sqrt(3*0.05**2)
    assert np.allclose(radial_distances, correct_distance)

    correct_vector = np.array((0.05, 0.05, 0.05))/correct_distance
    assert np.allclose(correct_vector, normalized_radial_vectors[0, :])
    assert np.allclose(correct_vector, normalized_radial_vectors[1, :])
    assert np.allclose(correct_vector, normalized_radial_vectors[2, :])


def test_radial_vector3():
    Lbox = 1.

    sat_x = [0.95, 0.95, 0.95]
    sat_y = [0.05, 0.95, 0.95]
    sat_z = [0.95, 0.95, 0.95]
    sat_hostid = [1, 0, 2]

    host_halo_x = [0.8, 0.05, 0.9]
    host_halo_y = [0.8, 0.1, 0.9]
    host_halo_z = [0.8, 0.85, 0.9]
    host_halo_id = [1, 2, 0]

    normalized_radial_vectors, radial_distances = calculate_satellite_radial_vector(
            sat_hostid, sat_x, sat_y, sat_z,
            host_halo_id, host_halo_x, host_halo_y, host_halo_z, Lbox)

    correct_distance0 = np.sqrt(0.15**2 + 0.25**2 + 0.15**2)
    assert np.allclose(radial_distances[0], correct_distance0)
    correct_distance1 = np.sqrt(3*0.05**2)
    assert np.allclose(radial_distances[1], correct_distance1)
    correct_distance2 = np.sqrt(0.1**2 + 0.15**2 + 0.1**2)
    assert np.allclose(radial_distances[2], correct_distance2)

    correct_vector0 = np.array((0.15, 0.25, 0.15))/correct_distance0
    assert np.allclose(correct_vector0, normalized_radial_vectors[0, :])
    correct_vector1 = np.array((0.05, 0.05, 0.05))/correct_distance1
    assert np.allclose(correct_vector1, normalized_radial_vectors[1, :])
    correct_vector2 = np.array((-0.1, -0.15, 0.1))/correct_distance2
    assert np.allclose(correct_vector2, normalized_radial_vectors[2, :])


def test_reposition_satellites1():
    r""" PBCs are irrelevant
    """
    nsats = int(1e3)
    Lbox = 1

    central_position = np.random.uniform(0.25, 0.75, nsats*3).reshape((nsats, 3))
    orig_radial_vector = np.random.uniform(-0.1, 0.1, nsats*3).reshape((nsats, 3))
    satellite_position = central_position + orig_radial_vector
    new_radial_vector = np.random.uniform(-0.1, 0.1, nsats*3).reshape((nsats, 3))
    new_satellite_position = reposition_satellites_from_radial_vectors(
        satellite_position, orig_radial_vector, new_radial_vector, Lbox)
    correct_satellite_position = central_position + new_radial_vector
    assert np.allclose(new_satellite_position, correct_satellite_position)


def test_reposition_satellites2():
    r""" PBCs are operative
    """
    Lbox = 1

    central_position = np.array((0.1, 0.1, 0.1)).reshape((1, 3))
    orig_radial_vector = np.array((0.2, -0.2, 0.0)).reshape((1, 3))
    satellite_position = np.array((0.3, 0.9, 0.1)).reshape((1, 3))
    new_radial_vector = np.array((0.01, 0.01, 0.01)).reshape((1, 3))

    new_satellite_position = reposition_satellites_from_radial_vectors(
        satellite_position, orig_radial_vector, new_radial_vector, Lbox)

    correct_satellite_position = central_position + new_radial_vector
    assert np.allclose(new_satellite_position, correct_satellite_position)
