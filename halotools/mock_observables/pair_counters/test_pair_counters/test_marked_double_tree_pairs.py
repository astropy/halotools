#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest 
slow = pytest.mark.slow

from copy import copy 

from ..pairs import wnpairs as pure_python_weighted_pairs
from ..pairs import xy_z_wnpairs as pure_python_xy_z_weighted_pairs
from ..marked_double_tree_pairs import marked_npairs, xy_z_marked_npairs
from ..marked_double_tree_helpers import _func_signature_int_from_wfunc
from ..double_tree_pairs import npairs

from ....custom_exceptions import HalotoolsError

__all__ = ['test_marked_npairs_periodic','test_marked_npairs_nonperiodic',\
           'test_xy_z_marked_npairs_periodic','test_xy_z_marked_npairs_nonperiodic',\
           'test_marked_npairs_wfuncs_signatures','test_marked_npairs_wfuncs_behavior']

#set up random points to test pair counters
np.random.seed(1)
Npts = 1000
random_sample = np.random.random((Npts,3))
period = np.array([1.0,1.0,1.0])
num_threads=2

#set up a regular grid of points to test pair counters
Npts2 = 10
epsilon = 0.001
gridx = np.linspace(0, 1-epsilon, Npts2)
gridy = np.linspace(0, 1-epsilon, Npts2)
gridz = np.linspace(0, 1-epsilon, Npts2)
xx, yy, zz = np.array(np.meshgrid(gridx, gridy, gridz))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()
grid_points = np.vstack([xx, yy, zz]).T

#set up random weights
ran_weights1 = np.random.random((Npts,1))
ran_weights2 = np.random.random((Npts,2))

def test_marked_npairs_periodic():
    """
    Function tests marked_npairs with periodic boundary conditions.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    result = marked_npairs(random_sample, random_sample, 
        rbins, period=period, weights1=ran_weights1, weights2=ran_weights1, wfunc=1)
    
    test_result = pure_python_weighted_pairs(random_sample, random_sample, rbins, 
        period=period, weights1=ran_weights1, weights2=ran_weights1)
    
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


def test_marked_npairs_nonperiodic():
    """
    Function tests marked_npairs with without periodic boundary conditions.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    result = marked_npairs(random_sample, random_sample,
        rbins, period=None, 
        weights1=ran_weights1, weights2=ran_weights1, wfunc=1)
    
    test_result = pure_python_weighted_pairs(random_sample, random_sample,
        rbins, period=None, weights1=ran_weights1, weights2=ran_weights1)
    
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"

def test_xy_z_marked_npairs_periodic():
    """
    Function tests xy_z_marked_npairs with periodic boundary conditions.
    """
    
    rp_bins = np.array([0.0,0.1,0.2,0.3])
    pi_bins = np.array([0.0,0.1,0.2,0.3])
    
    result = xy_z_marked_npairs(random_sample, random_sample, 
        rp_bins, pi_bins, period=period, weights1=ran_weights1, weights2=ran_weights1, wfunc=1)
    
    test_result = pure_python_xy_z_weighted_pairs(random_sample, random_sample, rp_bins, pi_bins,
        period=period, weights1=ran_weights1, weights2=ran_weights1)
    
    print(test_result)
    print(result)
    
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


def test_xy_z_marked_npairs_nonperiodic():
    """
    Function tests xy_z_marked_npairs with without periodic boundary conditions.
    """
    
    rp_bins = np.array([0.0,0.1,0.2,0.3])
    pi_bins = np.array([0.0,0.1,0.2,0.3])
    
    result = xy_z_marked_npairs(random_sample, random_sample,
        rp_bins, pi_bins, period=None, 
        weights1=ran_weights1, weights2=ran_weights1, wfunc=1)
    
    test_result = pure_python_xy_z_weighted_pairs(random_sample, random_sample,
        rp_bins, pi_bins, period=None, weights1=ran_weights1, weights2=ran_weights1)
    
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"

@slow
def test_marked_npairs_wfuncs_signatures():
    """ 
    Loop over all wfuncs and ensure that the wfunc signature is handled correctly.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    rmax = rbins.max()
    
    # Determine how many wfuncs have currently been implemented
    wfunc_index = 1
    while True:
        try:
            _ = _func_signature_int_from_wfunc(wfunc_index)
            wfunc_index += 1
        except HalotoolsError:
            break
    num_wfuncs = copy(wfunc_index)

    # Now loop over all all available wfunc indices
    for wfunc_index in xrange(1, num_wfuncs):
        signature = _func_signature_int_from_wfunc(wfunc_index)
        weights = np.random.random(Npts*signature).reshape(Npts, signature) - 0.5
        result = marked_npairs(random_sample, random_sample, rbins, period=period, 
            weights1=weights, weights2=weights, wfunc=wfunc_index, 
            approx_cell1_size = [rmax, rmax, rmax])

        with pytest.raises(HalotoolsError):
            signature = _func_signature_int_from_wfunc(wfunc_index) + 1
            weights = np.random.random(Npts*signature).reshape(Npts, signature) - 0.5
            result = marked_npairs(random_sample, random_sample, rbins, period=period, 
                weights1=weights, weights2=weights, wfunc=wfunc_index)

@slow
def test_marked_npairs_wfuncs_behavior():
    """ 
    Verify the behavior of a few wfunc-weighted counters by comparing pure python, 
    unmarked pairs to the returned result from a uniformly weighted set of points.  
    """
    
    error_msg = ("\nThe `test_marked_npairs_wfuncs_behavior` function performs \n"
        "non-trivial checks on the returned values of marked correlation functions\n"
        "calculated on a set of points with uniform weights.\n"
        "One such check failed.\n")
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    rmax = rbins.max()
    
    test_result = pure_python_weighted_pairs(grid_points, grid_points,
        rbins, period=period)

    # wfunc = 1
    weights = np.ones(Npts)*3
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=1, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 9.*test_result), error_msg

    # wfunc = 2
    weights = np.ones(Npts)*3
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=2, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 6.*test_result), error_msg

    # wfunc = 3
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=3, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 9.*test_result), error_msg

    weights = np.vstack([weights3, weights2]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=3, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 4.*test_result), error_msg

    # wfunc = 4
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=4, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 0), error_msg

    # wfunc = 5
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=5, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 0), error_msg

    # wfunc = 6
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=6, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 0), error_msg

    # wfunc = 7
    weights2 = np.ones(Npts)
    weights3 = np.zeros(Npts)-1

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=7, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == -test_result), error_msg

    # wfunc = 8
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=8, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 3*test_result), error_msg

    # wfunc = 9
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=9, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 3*test_result), error_msg

    # wfunc = 10
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=10, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == 0), error_msg

    weights2 = np.ones(Npts)
    weights3 = -np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(grid_points, grid_points, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=10, approx_cell1_size = [rmax, rmax, rmax])
    assert np.all(result == -3*test_result), error_msg


