#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..tpcf_one_two_halo_decomp import tpcf_one_two_halo_decomp
from ...custom_exceptions import *

import pytest
slow = pytest.mark.slow

__all__=['test_tpcf_one_two_halo_auto_periodic', 'test_tpcf_one_two_halo_cross_periodic']

#create toy data to test functions
Npts = 100
IDs1 = np.random.random_integers(0,10,Npts)
IDs2 = np.random.random_integers(0,10,Npts)
sample1 = np.random.random((Npts,3))
sample2 = np.random.random((Npts,3))
randoms = np.random.random((Npts*3,3))
period = np.array([1.0,1.0,1.0])
rbins = np.linspace(0,0.3,5)

def test_tpcf_one_two_halo_auto_periodic():
    """
    test the tpcf_one_two_halo autocorrelation with periodic boundary conditions
    """
    
    result = tpcf_one_two_halo_decomp(sample1, IDs1, rbins, sample2 = None, 
                                      randoms=None, period = period, 
                                      max_sample_size=int(1e4), estimator='Natural')
    
    assert len(result)==2, "wrong number of correlation functions returned."


def test_tpcf_one_two_halo_cross_periodic():
    """
    test the tpcf_one_two_halo cross-correlation with periodic boundary conditions
    """
    
    result = tpcf_one_two_halo_decomp(sample1, IDs1, rbins, sample2 = sample2,
                                      sample2_host_halo_id=IDs2,randoms=None,
                                      period = period, max_sample_size=int(1e4),
                                      estimator='Natural')
    
    assert len(result)==6, "wrong number of correlation functions returned."


