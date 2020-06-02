"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.config.paths import _find_home

from ..positional_marked_npairs_xy_z import positional_marked_npairs_xy_z
from ..npairs_xy_z import npairs_xy_z

from halotools.mock_observables.tests.cf_helpers import generate_3d_regular_mesh, generate_locus_of_3d_points
from rotations.vector_utilities import normalized_vectors, angles_between_list_of_vectors, elementwise_norm

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_1', 'test_2', 'test_3', 'test_4', 'test_threading', 'test_unweighted_counts')


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


def generate_aligned_vectors(npts, dim=2):
    """
    return a set of aligned vectors, all pointing in a random direction
    """

    vector = normalized_vectors(np.random.random(dim))
    vectors = np.tile(vector, npts).reshape((npts, dim))

    return vectors


def test_1():
    """
    test weighting function 1
    """

    # generate two locusts of points
    npts= 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)
    
    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))
  
    # calculate dot product between vectors1 and cluster 2
    rp = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2)
    pi = 0.9-0.1
    # s, vector between coords1 and cluster2
    sp = np.zeros((npts,2))
    sp[:,0] = 0.9 - coords1[:,0]
    sp[:,1] = 0.9 - coords1[:,1] 

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, sp)
    costheta = np.cos(angles) # dot product between vectors
    avg_costheta = np.mean(costheta)

    #define radial bins
    rp_bins = np.array([0.0, 0.1, rp+2.0*epsilon])
    pi_bins = np.array([0.0, 0.1, pi+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 3))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_xy_z(coords1, coords2, rp_bins, pi_bins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)
    
    msg = ("weighted counts do not match expected result given the weighting function")
    print(weighted_counts)
    assert np.isclose(weighted_counts[-1,-1], avg_costheta*counts[-1,-1], rtol=1.0/npts), msg


def test_2():
    """
    test weighting function 2
    """
    
    # generate two locusts of points
    npts= 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)
    
    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))
  
    # calculate dot product between vectors1 and cluster 2
    rp = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2)
    pi = 0.9-0.1
    # s, vector between coords1 and cluster2
    sp = np.zeros((npts,2))
    sp[:,0] = 0.9 - coords1[:,0]
    sp[:,1] = 0.9 - coords1[:,1] 

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, sp)
    avg_two_costheta_1 = np.mean(np.cos(2.0*angles))
    avg_two_costheta_2 = np.mean(2.0*np.cos(angles)*np.cos(angles) - 1.0)
    assert np.isclose(avg_two_costheta_1,avg_two_costheta_2) # test trig identify used in weighting function
    avg_two_costheta = avg_two_costheta_2

    #define radial bins
    rp_bins = np.array([0.0, 0.1, rp+2.0*epsilon])
    pi_bins = np.array([0.0, 0.1, pi+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 3))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_xy_z(coords1, coords2, rp_bins, pi_bins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=1)
    
    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1,-1], avg_two_costheta*counts[-1,-1], rtol=1.0/npts), msg


def test_3():
    """
    test weighting function 3
    """
    
    # generate two locusts of points
    npts= 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)
    
    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))
  
    # calculate dot product between vectors1 and cluster 2
    rp = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2)
    pi = 0.9-0.1
    # s, vector between coords1 and cluster2
    sp = np.zeros((npts,2))
    sp[:,0] = 0.9 - coords1[:,0]
    sp[:,1] = 0.9 - coords1[:,1] 

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, sp)
    avg_two_sintheta = np.mean(np.sin(2.0*angles))

    #define radial bins
    rp_bins = np.array([0.0, 0.1, rp+2.0*epsilon])
    pi_bins = np.array([0.0, 0.1, pi+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 3))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_xy_z(coords1, coords2, rp_bins, pi_bins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=1)
    
    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1,-1], avg_two_sintheta*counts[-1,-1], rtol=1.0/npts), msg


def test_4():
    """
    test weighting function 4
    """
    
    # generate two locusts of points
    npts= 100
    epsilon = 0.001
    # #cluster 1
    coords1 = generate_locus_of_3d_points(npts, 0.1, 0.1, 0.1, epsilon=epsilon)
    # cluster 2
    coords2 = generate_locus_of_3d_points(npts, 0.9, 0.9, 0.9, epsilon=epsilon)
    
    #generate orientation vectors for cluster 1
    vectors1 = generate_aligned_vectors(len(coords1))
  
    # calculate dot product between vectors1 and cluster 2
    rp = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2)
    pi = 0.9-0.1
    # s, vector between coords1 and cluster2
    sp = np.zeros((npts,2))
    sp[:,0] = 0.9 - coords1[:,0]
    sp[:,1] = 0.9 - coords1[:,1] 

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, sp)
    costheta_squared = np.cos(angles)*np.cos(angles) # dot product between vectors
    avg_costheta_squared = np.mean(costheta_squared)

    #define radial bins
    rp_bins = np.array([0.0, 0.1, rp+2.0*epsilon])
    pi_bins = np.array([0.0, 0.1, pi+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 3))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_xy_z(coords1, coords2, rp_bins, pi_bins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=1)
    
    msg = ("weighted counts do not match expected result given the weighting function")
    assert np.isclose(weighted_counts[-1,-1], avg_costheta_squared*counts[-1,-1], rtol=1.0/npts), msg


def test_threading():
    """
    test to make sure the result is the same with and without threading for each weighting function
    """
    
    npts = 100
    random_coords = np.random.random((npts,3))
    random_vectors = np.random.random((npts,3))*2.0-1.0

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.0, 0.3, 5)
    pi_bins = np.linspace(0.0, 0.3, 5)

    weights1 = np.ones((npts, 3))
    weights1[:,1] = random_vectors[:,0]
    weights1[:,2] = random_vectors[:,1]
    weights2 =  np.ones(npts)

    msg = ("counts do not match for different ``num_threads``.")

    weighted_counts_1, counts_1 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg

    weighted_counts_1, counts_1 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg

    weighted_counts_1, counts_1 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg

    weighted_counts_1, counts_1 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=1)
    weighted_counts_2, counts_2 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=3)

    assert np.allclose(weighted_counts_1, weighted_counts_2), msg
    assert np.allclose(counts_1, counts_2), msg


def test_unweighted_counts():
    """
    test to make sure the unweighted counts result is the same as npairs_3d
    """
    
    npts = 100
    random_coords = np.random.random((npts,3))
    random_vectors = np.random.random((npts,3))*2.0-1.0

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.0, 0.3, 5)
    pi_bins = np.linspace(0.0, 0.3, 5)

    weights1 = np.ones((npts, 3))
    weights1[:,1] = random_vectors[:,0]
    weights1[:,2] = random_vectors[:,1]
    weights2 =  np.ones(npts)

    weighted_counts_1, counts_1 = positional_marked_npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins,
                  period=period, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)
    counts_2 = npairs_xy_z(random_coords, random_coords, rp_bins, pi_bins, period=period,num_threads=3)
    
    msg = ('unweighted counts do no match npairs_3d result')
    assert np.allclose(counts_1, counts_2), msg







