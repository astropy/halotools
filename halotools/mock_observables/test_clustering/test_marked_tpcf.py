#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..marked_tpcf import marked_tpcf

import pytest
slow = pytest.mark.slow

__all__=['test_marked_tpcf_auto_periodic','test_marked_tpcf_auto_nonperiodic']

#create toy data to test functions
N_pts = 100
sample1 = np.random.random((N_pts,3))
sample2 = np.random.random((N_pts,3))
period = np.array([1.0,1.0,1.0])
rbins = np.linspace(0,0.3,5)

def test_marked_tpcf_auto_periodic():
    """
    test marked_tpcf auto correlation with periodic boundary conditions
    """
    
    wfunc = 1
    weights1 = np.random.random(N_pts)
    weights2 = np.random.random(N_pts)
    
    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,\
                         period=period, num_threads=1, wfunc=wfunc)
    
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_marked_tpcf_auto_nonperiodic():
    """
    test marked_tpcf auto correlation without periodic boundary conditions
    """
    
    wfunc = 1
    weights1 = np.random.random(N_pts)
    weights2 = np.random.random(N_pts)
    
    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,\
                         period=None, num_threads=1, wfunc=wfunc)
    
    assert result.ndim == 1, "More than one correlation function returned erroneously."
