#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys
from ..wp import wp

__all__=['test_wp_auto_nonperiodic','test_wp_auto_periodic','test_wp_cross_periodic',\
         'test_wp_cross_nonperiodic']

Npts = 100
sample1 = np.random.random((Npts,3))
sample2 = np.random.random((Npts,3))
randoms = np.random.random((Npts,3))
period = np.array([1.0,1.0,1.0])
rp_bins = np.linspace(0,0.3,5)
pi_bins = np.linspace(0,0.3,5)

def test_wp_auto_nonperiodic():
    """
    test wp autocorrelation without periodic boundary conditions
    """
    result = wp(sample1, rp_bins, pi_bins, sample2 = None, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural')
    
    print(result)
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_wp_auto_periodic():
    """
    test wp autocorrelation with periodic boundary conditions
    """
    result = wp(sample1, rp_bins, pi_bins, sample2 = None, 
                randoms=None, period = period, 
                max_sample_size=int(1e4), estimator='Natural')
    

    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_wp_cross_periodic():
    """
    test wp cross-correlation with periodic boundary conditions
    """
    result = wp(sample1, rp_bins, pi_bins, sample2 = sample2, 
                randoms=None, period = period, 
                max_sample_size=int(1e4), estimator='Natural')

    assert len(result)==3, "wrong number of correlations returned"
    assert result[0].ndim == 1, "dimension of auto incorrect"
    assert result[1].ndim == 1, "dimension of cross incorrect"
    assert result[2].ndim == 1, "dimension auto incorrect"

def test_wp_cross_nonperiodic():
    """
    test wp cross-correlation with periodic boundary conditions
    """
    result = wp(sample1, rp_bins, pi_bins, sample2 = sample2, 
                randoms=randoms, period = None, 
                max_sample_size=int(1e4), estimator='Natural')

    assert len(result)==3, "wrong number of correlations returned"
    assert result[0].ndim == 1, "dimension of auto incorrect"
    assert result[1].ndim == 1, "dimension of cross incorrect"
    assert result[2].ndim == 1, "dimension auto incorrect"

