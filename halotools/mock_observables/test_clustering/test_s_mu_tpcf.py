#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..s_mu_tpcf import s_mu_tpcf

####two point correlation function########################################################

def test_TPCF_auto_no_pbc():
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((1000,3))
    period = np.array([1,1,1])
    s_bins = np.linspace(0,0.3,5)
    mu_bins = np.linspace(0,1.0,10)
    
    result_1 = s_mu_tpcf(sample1, s_bins, mu_bins, sample2 = None, 
                       randoms=randoms, period = None, 
                       max_sample_size=int(1e4), estimator='Natural')
    
    assert result_1.ndim == 2, "correlation function returned has wrong dimension."


def test_TPCF_auto_w_pbc():
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((1000,3))
    period = np.array([1.0,1.0,1.0])
    s_bins = np.linspace(0,0.3,5)
    mu_bins = np.linspace(0,1.0,10)
    
    result_1 = s_mu_tpcf(sample1, s_bins, mu_bins, sample2 = None, 
                         randoms=randoms, period = period, 
                         max_sample_size=int(1e4), estimator='Natural')
    
    assert result_1.ndim == 2, "correlation function returned has wrong dimension."

