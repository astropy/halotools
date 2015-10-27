#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..marked_tpcf import marked_tpcf

import pytest
slow = pytest.mark.slow

__all__=['test_TPCF_auto', 'test_wfuncs']

####two point correlation function########################################################

def test_TPCF_auto():
    
    N_pts = 100
    
    sample1 = np.random.random((N_pts,3))
    sample2 = np.random.random((N_pts,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    wfunc = 1
    weights1 = np.random.random(N_pts)
    weights2 = np.random.random(N_pts)
    aux1 = np.random.random(N_pts)
    aux2 = np.random.random(N_pts)
    
    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,\
                         period=None, num_threads=1,\
                         aux1=None, aux2=None, wfunc=wfunc)
    
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_wfuncs():
    
    N_pts = 1000
    
    sample1 = np.random.random((N_pts,3))
    sample2 = np.random.random((N_pts,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    weights1 = np.random.random(N_pts)
    weights2 = np.random.random(N_pts)
    aux1 = np.random.random(N_pts)
    aux2 = np.random.random(N_pts)
    
    result = np.zeros((10-1,4))
    for i in range(1,10):
        print(i)
        result[i-1,:] = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,\
                                    period=None, num_threads=1,\
                                    aux1=None, aux2=None, wfunc=i)
        assert result[i-1,:].ndim == 1, "More than one correlation function returned erroneously."


