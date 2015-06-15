#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import sys
from ..clustering import redshift_space_tpcf

__all__=['test_rs_tpcf_auto','test_rs_tpcf_auto_periodic','test_rs_tpcf_cross_periodic',]

####two point correlation function########################################################

def test_wp_auto():
    Npts=100
    
    sample1 = np.random.random((Npts,3))
    sample2 = np.random.random((Npts,3))
    randoms = np.random.random((Npts,3))
    period = np.array([1,1,1])
    rp_bins = np.linspace(0,0.5,5)
    pi_bins = np.linspace(0,0.5,5)
    
    #with randoms
    result = redshift_space_tpcf(sample1, rp_bins, pi_bins, sample2 = None, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural')

    assert result.ndim == 2, "More than one correlation function returned erroneously."


def test_wp_auto_periodic():
    Npts=100
    
    sample1 = np.random.random((Npts,3))
    sample2 = np.random.random((Npts,3))
    randoms = np.random.random((Npts,3))
    period = np.array([1,1,1])
    rp_bins = np.linspace(0,0.5,5)
    pi_bins = np.linspace(0,0.5,5)
    
    #with randoms
    result = redshift_space_tpcf(sample1, rp_bins, pi_bins, sample2 = None, 
                  randoms=randoms, period = period, 
                  max_sample_size=int(1e4), estimator='Natural')
    
    assert result.ndim == 2, "More than one correlation function returned erroneously."


def test_wp_cross_periodic():
    Npts=100
    
    sample1 = np.random.random((Npts,3))
    sample2 = np.random.random((Npts,3))
    randoms = np.random.random((Npts,3))
    period = np.array([1,1,1])
    rp_bins = np.linspace(0,0.5,5)
    pi_bins = np.linspace(0,0.5,5)
    
    #with randoms
    result = redshift_space_tpcf(sample1, rp_bins, pi_bins, sample2 = sample2, 
                  randoms=randoms, period = period, 
                  max_sample_size=int(1e4), estimator='Natural')
    
    assert len(result)==3, "wrong number of correlations returned"
    assert result[0].ndim == 2, "dimension of auto incorrect"
    assert result[1].ndim == 2, "dimension of cross incorrect"
    assert result[2].ndim == 2, "dimension auto incorrect"


