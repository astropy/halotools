#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys
import pytest 

from ..tpcf_jackknife import tpcf_jackknife
from ..tpcf import tpcf

__all__=['test_tpcf_jackknife']

@pytest.mark.slow
def test_tpcf_jackknife():
    
    Npts=100
    sample1 = np.random.random((Npts,3))
    randoms = np.random.random((Npts*10,3))
    period = np.array([1.0,1.0,1.0])
    Lbox = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0.0,0.3,5).astype(float)
    
    result_1,err = tpcf_jackknife(sample1, randoms, rbins, Nsub=5, period = period, num_threads=1)
    result_2 = tpcf(sample1, rbins, randoms=randoms, period = period, num_threads=1)
    
    print(result_1)
    print(err)
    print(result_2)
    assert np.allclose(result_1,result_2,rtol=1e-09), "correlation functions do not match"


@pytest.mark.slow
def test_tpcf_jackknife_cov_matrix():
    
    Npts=100
    sample1 = np.random.random((Npts,3))
    randoms = np.random.random((Npts*10,3))
    period = np.array([1.0,1.0,1.0])
    Lbox = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0.0,0.3,5)
    nbins = len(rbins)-1
    
    result_1,err = tpcf_jackknife(sample1, randoms, rbins, Nsub=5, period = period, num_threads=1)
    
    print(err)
    assert np.shape(err)==(nbins,nbins), "correlation functions do not match"