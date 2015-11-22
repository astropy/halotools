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

#set up random points to test pair counters
np.random.seed(1)
Npts = 1000
random_sample = np.random.random((Npts,3))
period = np.array([1.0,1.0,1.0])


def test_npairs_periodic():
    """
    test npairs with periodic boundary conditons.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    result = npairs(random_sample, random_sample, rbins, period=period, verbose=True,\
                    num_threads=1)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rbins),), msg
    
    test_result = simp_npairs(random_sample, random_sample, rbins, period=period)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result==result), msg


def test_npairs_nonperiodic():
    """
    test npairs without periodic boundary conditons.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    result = npairs(random_sample, random_sample, rbins, period=None)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rbins),), msg
    
    test_result = simp_npairs(random_sample, random_sample, rbins, period=None)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result==result), msg


def test_xy_z_npairs_periodic():
    """
    test xy_z_npairs with periodic boundary conditons.
    """
    
    rp_bins = np.arange(0,0.31,0.1)
    pi_bins = np.arange(0,0.31,0.1)
    
    result = xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins, period=period)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rp_bins),len(pi_bins)), msg
    
    test_result = simp_xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins,\
                                   period=period)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_xy_z_npairs_nonperiodic():
    """
    test xy_z_npairs without periodic boundary conditons.
    """
    
    rp_bins = np.arange(0,0.31,0.1)
    pi_bins = np.arange(0,0.31,0.1)
    
    result = xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins, period=None)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(rp_bins),len(pi_bins)), msg
    
    test_result = simp_xy_z_npairs(random_sample, random_sample, rp_bins, pi_bins,\
                                   period=None)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_s_mu_npairs_periodic():
    """
    test s_mu_npairs with periodic boundary conditons.
    """
    
    s_bins = np.array([0.0,0.1,0.2,0.3])
    N_mu_bins=100
    mu_bins = np.linspace(0,1.0,N_mu_bins)
    Npts = len(random_sample)
    
    result = s_mu_npairs(random_sample, random_sample, s_bins, mu_bins, period=period)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(s_bins),N_mu_bins), msg
    
    result = np.diff(result,axis=1)
    result = np.sum(result, axis=1)+ Npts
    
    test_result = npairs(random_sample, random_sample, s_bins, period=period)
    
    print(test_result)
    print(result)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_s_mu_npairs_nonperiodic():
    """
    test s_mu_npairs without periodic boundary conditons.
    """
    
    s_bins = np.array([0.0,0.1,0.2,0.3])
    N_mu_bins=100
    mu_bins = np.linspace(0,1.0,N_mu_bins)
    
    result = s_mu_npairs(random_sample, random_sample, s_bins, mu_bins)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(s_bins),N_mu_bins), msg
    
    result = np.diff(result,axis=1)
    result = np.sum(result, axis=1)+ Npts
    
    test_result = npairs(random_sample, random_sample, s_bins)
    
    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_jnpairs_periodic():
    """
    test jnpairs with periodic boundary conditons.
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
                     weights1=weights1, weights2=weights1)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(N_jsamples+1,len(rbins)), msg


def test_jnpairs_nonperiodic():
    """
    test jnpairs without periodic boundary conditons.
    """
    
    rbins = np.array([0.0,0.1,0.2,0.3])
    
    #define the jackknife sample labels
    Npts = len(random_sample)
    N_jsamples=10
    jtags1 = np.sort(np.random.random_integers(1, N_jsamples, size=Npts))
    
    #define weights
    weights1 = np.random.random(Npts)
    
    result = jnpairs(random_sample, random_sample, rbins, period=None,\
                     jtags1=jtags1, jtags2=jtags1, N_samples=10,\
                     weights1=weights1, weights2=weights1)
    
    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(N_jsamples+1,len(rbins)), msg


