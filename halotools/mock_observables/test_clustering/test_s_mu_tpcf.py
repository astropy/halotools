#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..clustering import tpcf, s_mu_tpcf

####two point correlation function########################################################

def test_TPCF_auto():
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    s_bins = np.linspace(0,0.5,5)
    mu_bins = np.linspace(0,1.0,10)
    
    
    result = s_mu_tpcf(sample1, s_bins, mu_bins, sample2 = None, 
                       randoms=randoms, period = None, 
                       max_sample_size=int(1e4), estimator='Natural')
    
    assert result.ndim == 2, "correlation function returned has wrong dimension."