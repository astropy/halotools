#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
#load pair counters
from ..double_tree_pairs import npairs, jnpairs, xy_z_npairs, s_mu_npairs
#load comparison simple pair counters
from ..pairs import npairs as simp_npairs
from ..pairs import wnpairs as simp_wnpairs
from ..pairs import xy_z_npairs as simp_xy_z_npairs


from ....custom_exceptions import HalotoolsError

import pytest
slow = pytest.mark.slow

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

grid_jackknife_spacing = 0.5
grid_jackknife_ncells = int(1/grid_jackknife_spacing)
ix = np.floor(gridx/grid_jackknife_spacing).astype(int)
iy = np.floor(gridy/grid_jackknife_spacing).astype(int)
iz = np.floor(gridz/grid_jackknife_spacing).astype(int)
ixx, iyy, izz = np.array(np.meshgrid(ix, iy, iz))
ixx = ixx.flatten()
iyy = iyy.flatten()
izz = izz.flatten()
grid_indices = np.ravel_multi_index([ixx, iyy, izz], 
    [grid_jackknife_ncells, grid_jackknife_ncells, grid_jackknife_ncells])
grid_indices += 1

def test_npairs_periodic():
    """
    Function tests npairs with periodic boundary conditions.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    result = npairs(random_sample, random_sample, rbins, period=period,\
                    num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rbins),), msg
    
    test_result = simp_npairs(random_sample, random_sample, rbins, period=period)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result==result), msg


def test_npairs_nonperiodic():
    """
    test npairs without periodic boundary conditions.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    result = npairs(random_sample, random_sample, rbins, period=None,\
                    num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rbins),), msg
    
    test_result = simp_npairs(random_sample, random_sample, rbins, period=None)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result==result), msg


def test_xy_z_npairs_periodic():
    """
    test xy_z_npairs with periodic boundary conditions.
    """
    
    rp_bins = np.arange(0,0.31,0.1)
    pi_bins = np.arange(0,0.31,0.1)
    
    result = xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins, period=period,\
                         num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rp_bins),len(pi_bins)), msg
    
    test_result = simp_xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins,\
                                   period=period)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_xy_z_npairs_nonperiodic():
    """
    test xy_z_npairs without periodic boundary conditions.
    """
    
    rp_bins = np.arange(0,0.31,0.1)
    pi_bins = np.arange(0,0.31,0.1)
    
    result = xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins, period=None,\
                         num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rp_bins),len(pi_bins)), msg
    
    test_result = simp_xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins,\
                                   period=None)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_s_mu_npairs_periodic():
    """
    test s_mu_npairs with periodic boundary conditions.
    """
    
    s_bins = np.array([0.0,0.1,0.2,0.3])
    N_mu_bins=100
    mu_bins = np.linspace(0,1.0,N_mu_bins)
    Npts = len(random_sample)
    
    result = s_mu_npairs(random_sample, random_sample, s_bins, mu_bins, period=period,\
                         num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(s_bins),N_mu_bins), msg
    
    result = np.diff(result,axis=1)
    result = np.sum(result, axis=1)+ Npts
    
    test_result = npairs(random_sample, random_sample, s_bins, period=period,\
                         num_threads=num_threads)
    
    print(test_result)
    print(result)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_s_mu_npairs_nonperiodic():
    """
    test s_mu_npairs without periodic boundary conditions.
    """
    
    s_bins = np.array([0.0,0.1,0.2,0.3])
    N_mu_bins=100
    mu_bins = np.linspace(0,1.0,N_mu_bins)
    
    result = s_mu_npairs(random_sample, random_sample, s_bins, mu_bins,\
                         num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(s_bins),N_mu_bins), msg
    
    result = np.diff(result,axis=1)
    result = np.sum(result, axis=1)+ Npts
    
    test_result = npairs(random_sample, random_sample, s_bins, num_threads=num_threads)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_jnpairs_periodic():
    """
    test jnpairs with periodic boundary conditions.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    #define the jackknife sample labels
    Npts = len(random_sample)
    N_jsamples=10
    jtags1 = np.sort(np.random.random_integers(1, N_jsamples, size=Npts))
    
    #define weights
    weights1 = np.random.random(Npts)
    
    result = jnpairs(random_sample, random_sample, rbins, period=period,\
                     jtags1=jtags1, jtags2=jtags1, N_samples=10,\
                     weights1=weights1, weights2=weights1, num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(N_jsamples+1,len(rbins)), msg

    # Now verify that when computing jackknife pairs on a regularly spaced grid, 
    # the counts in all subvolumes are identical

    grid_result = jnpairs(grid_points, grid_points, rbins, period=period,
        jtags1=grid_indices, jtags2=grid_indices, N_samples=grid_jackknife_ncells**3,
        num_threads=num_threads)

    for icell in xrange(1, grid_jackknife_ncells**3-1):
        assert np.all(grid_result[icell, :] == grid_result[icell+1, :])


def test_jnpairs_nonperiodic():
    """
    test jnpairs without periodic boundary conditions.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    #define the jackknife sample labels
    Npts = len(random_sample)
    N_jsamples=10
    jtags1 = np.sort(np.random.random_integers(1, N_jsamples, size=Npts))
    
    #define weights
    weights1 = np.random.random(Npts)
    
    result = jnpairs(random_sample, random_sample, rbins, period=period,\
                     jtags1=jtags1, jtags2=jtags1, N_samples=10,\
                     weights1=weights1, weights2=weights1, num_threads=num_threads)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(N_jsamples+1,len(rbins)), msg


