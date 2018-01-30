"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.config.paths import _find_home

from ..pairs import wnpairs as pure_python_weighted_pairs
from ..positional_marked_npairs_3d import positional_marked_npairs_3d

from halotools.mock_observables.tests.cf_helpers import generate_3d_regular_mesh, generate_locus_of_3d_points
from halotools.utils import normalized_vectors, angles_between_list_of_vectors

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

error_msg = ("\nThe `test_positional_marked_npairs_wfuncs_behavior` function performs \n"
    "non-trivial checks on the returned values of marked correlation functions\n"
    "calculated on a set of points with uniform weights.\n"
    "One such check failed.\n")

__all__ = ('test_1', )


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

    vector = normalized_vectors(np.random.random(3))
    vectors = np.tile(vector, npts).reshape((npts, 3))

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
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts,3))
    s[:,0] = 0.9 - coords1[:,0]
    s[:,1] = 0.9 - coords1[:,1] 
    s[:,2] = 0.9 - coords1[:,2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    costheta = np.cos(angles) # dot product between vectors
    avg_costheta = np.mean(costheta)

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights1[:,3] = vectors1[:,2]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=1, num_threads=1)
    
    assert np.isclose(weighted_counts[-1], avg_costheta*counts[-1], rtol=1.0/npts)


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
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts,3))
    s[:,0] = 0.9 - coords1[:,0]
    s[:,1] = 0.9 - coords1[:,1] 
    s[:,2] = 0.9 - coords1[:,2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    avg_two_costheta_1 = np.mean(np.cos(2.0*angles))
    avg_two_costheta_2 = np.mean(2.0*np.cos(angles)*np.cos(angles) - 1.0)
    assert np.isclose(avg_two_costheta_1,avg_two_costheta_2) # test trig identify used in weighting function
    avg_two_costheta = avg_two_costheta_2

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights1[:,3] = vectors1[:,2]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=2, num_threads=1)
    
    assert np.isclose(weighted_counts[-1], avg_two_costheta*counts[-1], rtol=1.0/npts)


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
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts,3))
    s[:,0] = 0.9 - coords1[:,0]
    s[:,1] = 0.9 - coords1[:,1] 
    s[:,2] = 0.9 - coords1[:,2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    avg_two_sintheta = np.mean(np.sin(2.0*angles))

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights1[:,3] = vectors1[:,2]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=3, num_threads=1)
    
    assert np.isclose(weighted_counts[-1], avg_two_sintheta*counts[-1], rtol=1.0/npts)



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
    r = np.sqrt((0.9-0.1)**2 + (0.9-0.1)**2 + (0.9-0.1)**2)
    # s, vector between coords1 and cluster2
    s = np.zeros((npts,3))
    s[:,0] = 0.9 - coords1[:,0]
    s[:,1] = 0.9 - coords1[:,1] 
    s[:,2] = 0.9 - coords1[:,2]

    #calculate dot product between orientation and direction between cluster 1 and 2
    angles = angles_between_list_of_vectors(vectors1, s)
    costheta_squared = np.cos(angles)*np.cos(angles) # dot product between vectors
    avg_costheta_squared = np.mean(costheta_squared)

    #define radial bins
    rbins = np.array([0.0, 0.1, r+2.0*epsilon])

    # define weights appropiate for weighting function
    weights1 = np.ones((npts, 4))
    weights1[:,1] = vectors1[:,0]
    weights1[:,2] = vectors1[:,1]
    weights1[:,3] = vectors1[:,2]
    weights2 =  np.ones(npts)

    #calculate weighted counts
    weighted_counts, counts = positional_marked_npairs_3d(coords1, coords2, rbins,
                  period=None, weights1=weights1, weights2=weights2,
                  weight_func_id=4, num_threads=1)
    
    assert np.isclose(weighted_counts[-1], avg_costheta_squared*counts[-1], rtol=1.0/npts)




