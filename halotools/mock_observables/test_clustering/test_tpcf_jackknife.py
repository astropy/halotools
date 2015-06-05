#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import sys
from ..clustering import tpcf_jackknife, tpcf

__all__=['test_tpcf_jackknife']


def test_tpcf_jackknife():
    
    Npts=1000
    sample1 = np.random.random((Npts,3))
    randoms = np.random.random((Npts*10,3))
    period = np.array([1,1,1])
    Lbox = np.array([1,1,1])
    rbins = np.linspace(0.0,0.1,5)
    
    result_1,err = tpcf_jackknife(sample1, randoms, rbins, Nsub=5, Lbox=Lbox, period = period, N_threads=1)
    result_2 = tpcf(sample1, rbins, randoms=randoms, period = period, N_threads=1)
    
    print(result_1)
    print(err)
    print(result_2)
    assert np.all(result_1==result_2), "correlation functions do not match"