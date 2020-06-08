"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.config.paths import _find_home

from .pure_python_positonal_marked_npairs import cos2theta_pairs
from ..positional_marked_npairs_3d import positional_marked_npairs_3d
from ..npairs_3d import npairs_3d

from halotools.mock_observables.tests.cf_helpers import generate_3d_regular_mesh, generate_locus_of_3d_points
from halotools.utils.vector_utilities import normalized_vectors, angles_between_list_of_vectors

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_1', 'test_2', 'test_3', 'test_4', 'test_threading', 'test_unweighted_counts', 'test_compare_to_pure_python_result')


def generate_interlacing_grids(npts_per_dim, period=1.0):
    """
    return two sets of interlaced points on a grid
    """

    dmin, dmax = 0.0, period

    dx = (dmax - dmin)/float(npts_per_dim)
    npts_mesh1 = npts_per_dim**3

    mesh1_points = generate_3d_regular_mesh(npts_per_dim, dmin=dmin, dmax=dmax)

    mesh2_points = mesh1_points + dx/2.
    npts_mesh2 = mesh2_points.shape[0]

    return mesh1_points, mesh2_points


def generate_aligned_vectors(npts, dim=3):
    """
    return a set of aligned vectors, all pointing in a random direction
    """

    vector = normalized_vectors(np.random.random(dim))
    vectors = np.tile(vector, npts).reshape((npts, dim))

    return vectors


def compute_limiting(coords1, coords2, npts, rbins, weight_func_id, alignment):
    # generate vectors parallel, perpendicular or antiparallel to separation vector (note that this should be two point clouds rather than a range of particle positions)
    diff_coords = (coords2 - coords1)
    norm_value = np.sqrt(diff_coords[:, 0]**2 + diff_coords[:, 1]**2 + diff_coords[:, 2]**2)
    if alignment == "parallel":
        vector = (diff_coords.T/norm_value).T
    elif alignment == "perpendicular":
        vector = np.cross(diff_coords, [0, 0, 1])
    elif alignment == "antiparallel":
        vector = -(diff_coords.T/norm_value).T

    weights1 = np.ones((npts, 4))
    weights1[:, 1] = vector[:, 0]
    weights1[:, 2] = vector[:, 1]
    weights1[:, 3] = vector[:, 2]
    weights2 = np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
              period=None, weights1=weights1, weights2=weights2,
              weight_func_id=weight_func_id, num_threads=1)
    return weighted_counts, counts


def test_limits():
    """
    test limiting cases for angles
    """
    # generate two locusts of points
    npts = 100
    epsilon = 0.000
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)

    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))

    # calculate dot product between vectors1 and cluster 2
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # weighting 1
    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 1, alignment="parallel")
    msg = ("weighted counts do not match expected parallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 1.0*counts[-1], rtol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 1, alignment="perpendicular")
    msg = ("weighted counts do not match expected perpendicular result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 0.0*counts[-1], atol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 1, alignment="antiparallel")
    msg = ("weighted counts do not match expected antiparallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], -1.0*counts[-1], rtol=1.0/npts), msg

    # weighting 2
    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 2, alignment="parallel")
    msg = ("weighted counts do not match expected parallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 1.0*counts[-1], rtol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 2, alignment="perpendicular")
    msg = ("weighted counts do not match expected perpendicular result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], -1.0*counts[-1], atol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 2, alignment="antiparallel")
    msg = ("weighted counts do not match expected antiparallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 1.0*counts[-1], rtol=1.0/npts), msg

   # weighting 3
    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 3, alignment="parallel")
    msg = ("weighted counts do not match expected parallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 0.0*counts[-1], atol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 3, alignment="perpendicular")
    msg = ("weighted counts do not match expected perpendicular result given the weighting function"+ str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 0.0*counts[-1], atol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 3, alignment="antiparallel")
    msg = ("weighted counts do not match expected antiparallel result given the weighting function"+ str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 0.0*counts[-1], atol=1.0/npts), msg

    #weighting 4
    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 4, alignment="parallel")
    msg = ("weighted counts do not match expected parallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 1.0*counts[-1], rtol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 4, alignment="perpendicular")
    msg = ("weighted counts do not match expected perpendicular result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 0.0*counts[-1], atol=1.0/npts), msg

    weighted_counts, counts = compute_limiting(coords1, coords2, npts, rbins, 4, alignment="antiparallel")
    msg = ("weighted counts do not match expected antiparallel result given the weighting function" + str(weighted_counts[-1]) + " " + str(counts[-1]))
    assert np.isclose(weighted_counts[-1], 1.0*counts[-1], rtol=1.0/npts), msg


def test_1():
    """
    test weighting function 1
    """

    # generate two locusts of points
    npts = 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)

    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))

    # calculate dot product between vectors1 and cluster 2
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts, 3))
    s[:, 0] = 0.9 - coords1[:, 0]
    s[:, 1] = 0.9 - coords1[:, 1]
    s[:, 2] = 0.9 - coords1[:, 2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    costheta = np.cos(angles)  # dot product between vectors
    avg_costheta = np.mean(costheta)

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:, 1] = vectors1[:, 0]
    weights1[:, 2] = vectors1[:, 1]
    weights1[:, 3] = vectors1[:, 2]
    weights2 = np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)

    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1], avg_costheta*counts[-1], rtol=1.0/npts), msg


def test_2():
    """
    test weighting function 2
    """

    # generate two locusts of points
    npts = 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)

    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))

    # calculate dot product between vectors1 and cluster 2
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts, 3))
    s[:, 0] = 0.9 - coords1[:, 0]
    s[:, 1] = 0.9 - coords1[:, 1]
    s[:, 2] = 0.9 - coords1[:, 2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    avg_two_costheta_1 = np.mean(np.cos(2.0*angles))
    avg_two_costheta_2 = np.mean(2.0*np.cos(angles)*np.cos(angles) - 1.0)
    assert np.isclose(avg_two_costheta_1,avg_two_costheta_2)  # test trig identify used in weighting function
    avg_two_costheta = avg_two_costheta_2

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:, 1] = vectors1[:, 0]
    weights1[:, 2] = vectors1[:, 1]
    weights1[:, 3] = vectors1[:, 2]
    weights2 = np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=1)

    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1], avg_two_costheta*counts[-1], rtol=1.0/npts), msg


def test_3():
    """
    test weighting function 3
    """

    # generate two locusts of points
    npts = 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)

    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))

    # calculate dot product between vectors1 and cluster 2
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts, 3))
    s[:, 0] = 0.9 - coords1[:, 0]
    s[:, 1] = 0.9 - coords1[:, 1]
    s[:, 2] = 0.9 - coords1[:, 2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    avg_two_sintheta = np.mean(np.sin(2.0*angles))

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:, 1] = vectors1[:, 0]
    weights1[:, 2] = vectors1[:, 1]
    weights1[:, 3] = vectors1[:, 2]
    weights2 = np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=1)

    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1], avg_two_sintheta*counts[-1], rtol=1.0/npts), msg


def test_4():
    """
    test weighting function 4
    """

    # generate two locusts of points
    npts = 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)

    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))

    # calculate dot product between vectors1 and cluster 2
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts, 3))
    s[:, 0] = 0.9 - coords1[:, 0]
    s[:, 1] = 0.9 - coords1[:, 1]
    s[:, 2] = 0.9 - coords1[:, 2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    costheta_squared = np.cos(angles)*np.cos(angles)  # dot product between vectors
    avg_costheta_squared = np.mean(costheta_squared)

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:, 1] = vectors1[:, 0]
    weights1[:, 2] = vectors1[:, 1]
    weights1[:, 3] = vectors1[:, 2]
    weights2 = np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=1)

    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1], avg_costheta_squared*counts[-1], rtol=1.0/npts), msg


def test_randoms():
    """
    test for randomly distributed points and orientations
    """

    # generate two locusts of points
    npts = 200
    epsilon = 0.3
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.0, 0.0, 0.0, epsilon=epsilon)
    coords2 = generate_locus_of_3d_points(npts, 0.0, 0.0, 0.0, epsilon=epsilon)

    # generate orientation vectors for cluster 1
    vectors1 = normalized_vectors(np.random.random((npts, 3))*2. - 1. )

    #define radial bins
    rbins = np.array([0.0, 0.1, 2.0*epsilon])
    s = np.zeros((npts, 3))

    s[:, 0] = coords2[:, 0] - coords1[:, 0]
    s[:, 1] = coords2[:, 1] - coords1[:, 1]
    s[:, 2] = coords2[:, 2] - coords1[:, 2]

    dist_a = s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:, 1] = vectors1[:, 0]
    weights1[:, 2] = vectors1[:, 1]
    weights1[:, 3] = vectors1[:, 2]
    weights2 =  np.ones(npts)

    # weighting 1
    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                        period=None, weights1=weights1, weights2=weights2,
                        weight_func_id=1, num_threads=1)

    msg = ("weighted counts do not match expected result given the weighting function" + " " + str(np.max(dist_a)) )
    assert np.isclose(weighted_counts[-1], 0.0*counts[-1], atol=4000), msg


def test_weighting_implementation():
    """
    test that indexing is correct for weighting
    """

    # generate two locusts of points
    npts = 100
    epsilon = 0.05
    # cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)

    # generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))

    # generate a random index value to check for each cluster
    idx = np.random.randint(npts)
    idx2 = np.random.randint(npts)

    # calculate dot product between vectors1 and cluster 2
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((3))
    s[0] = coords2[idx2, 0] - coords1[idx, 0]
    s[1] = coords2[idx2, 1] - coords1[idx, 1]
    s[2] = coords2[idx2, 2] - coords1[idx, 2]

    # calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1[idx], s)
    costheta = np.cos(angles)  # dot product between vectors

    idx_costheta = costheta

    # define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.zeros((npts, 4))
    weights1[idx] = 1.0
    weights1[:, 1] = vectors1[:, 0]
    weights1[:, 2] = vectors1[:, 1]
    weights1[:, 3] = vectors1[:, 2]
    weights2 = np.zeros(npts)
    weights2[idx2] = 1.0

    # calculate weighted counts

    # weighting 1
    # calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                            period=None, weights1=weights1, weights2=weights2,
                            weight_func_id=1, num_threads=1)

    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1], idx_costheta, rtol=0.01/npts), msg


def test_threading():
    """
    test to make sure the result is the same with and without threading for each weighting function
    """

    npts = 100
    random_coords = np.random.random((npts, 3))
    random_vectors = np.random.random((npts, 3))*2.0-1.0

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.0, 0.3, 5)

    weights1 = np.ones((npts, 4))
    weights1[:, 1] = random_vectors[:, 0]
    weights1[:, 2] = random_vectors[:, 1]
    weights1[:, 3] = random_vectors[:, 2]
    weights2 = np.ones(npts)

    msg = ("counts do not match for different ``num_threads``.")

    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg

    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg

    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg

    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg


def test_unweighted_counts():
    """
    test to make sure the unweighted counts result is the same as npairs_3d
    """

    npts = 100
    random_coords = np.random.random((npts, 3))
    random_vectors = np.random.random((npts, 3))*2.0-1.0

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.0, 0.3, 5)

    weights1 = np.ones((npts, 4))
    weights1[:, 1] = random_vectors[:, 0]
    weights1[:, 2] = random_vectors[:, 1]
    weights1[:, 3] = random_vectors[:, 2]
    weights2 = np.ones(npts)

    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords, random_coords, rbins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)
    counts_2 = npairs_3d(random_coords, random_coords, rbins, period=period, num_threads=3)

    msg = ('unweighted counts do no match npairs_3d result')
    assert np.allclose(counts_1, counts_2), msg


def test_compare_to_pure_python_result():
    """
    test to compare pair counter to a pure python implemnetation.
    """

    npts = 4
    random_coords = np.random.random((npts, 3))
    random_vectors = normalized_vectors(np.random.random((npts, 3))*2.0-1.0)

    weights1 = np.ones((npts, 4))
    weights1[:, 1] = random_vectors[:, 0]
    weights1[:, 2] = random_vectors[:, 1]
    weights1[:, 3] = random_vectors[:, 2]
    weights2 = np.ones(npts)

    # define radial bins
    rbins=np.linspace(0.0, 0.3, 5)

    #with PBCs
    period = np.array([1.0, 1.0, 1.0])
    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords , random_coords , rbins,
                            period=period, weights1=weights1, weights2=weights2,
                            weight_func_id=4, num_threads=1)

    weighted_counts_2, counts_2 = cos2theta_pairs(random_coords , random_vectors, random_coords , rbins, period=period)

    msg = ('result does not match pure python result')
    assert np.allclose(weighted_counts_1, weighted_counts_2), msg

    #without PBCs
    period = None
    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords , random_coords , rbins,
                            period=period, weights1=weights1, weights2=weights2,
                            weight_func_id=4, num_threads=1)

    weighted_counts_2, counts_2 = cos2theta_pairs(random_coords , random_vectors, random_coords , rbins, period=period)

    msg = ('result does not match pure python result')
    assert np.allclose(weighted_counts_1, weighted_counts_2), msg

    #without threads
    period = np.array([1.0, 1.0, 1.0])
    weighted_counts_1, counts_1 = positional_marked_npairs_3d(random_coords , random_coords , rbins,
                            period=period, weights1=weights1, weights2=weights2,
                            weight_func_id=4, num_threads=3)

    weighted_counts_2, counts_2 = cos2theta_pairs(random_coords , random_vectors, random_coords , rbins, period=period)

    msg = ('result does not match pure python result')
    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
