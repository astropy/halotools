#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..tpcf import tpcf
from ...custom_exceptions import *

import pytest
slow = pytest.mark.slow

__all__=['test_TPCF_auto', 'test_TPCF_cross', 'test_TPCF_estimators',\
         'test_TPCF_sample_size_limit',\
         'test_TPCF_randoms', 'test_TPCF_period_API']

"""
Note that these are almost all unit-tests.  Non tirival tests are a little heard to think
of here.
"""

def test_TPCF_auto():
    """
    test the auto-correlation functionality
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    #with randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural')
    assert result.ndim == 1, "More than one correlation function returned erroneously."
    
    #with out randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=None, period = period, 
                  max_sample_size=int(1e4), estimator='Natural')
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_TPCF_cross():
    """
    test the cross-correlation functionality
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    #with randoms
    result = tpcf(sample1, rbins, sample2 = sample2, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural', do_auto=False)
    assert result.ndim == 1, "More than one correlation function returned erroneously."
    
    #with out randoms
    result = tpcf(sample1, rbins, sample2 = sample2, 
                  randoms=None, period = period, 
                  max_sample_size=int(1e4), estimator='Natural', do_auto=False)
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_TPCF_estimators():
    """
    test the different estimators functionality
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.3,5)
    
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Natural')
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Davis-Peebles')
    result_3 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Hewett')
    result_4 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Hamilton')
    result_5 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Landy-Szalay')
                                            
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of  correlation functions returned erroneously."
    assert len(result_3)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_4)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_5)==3, "wrong number of correlation functions returned erroneously."


def test_TPCF_sample_size_limit():
    """
    test the sample size limit functionality functionality
    """
    
    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((1000,3))
    randoms = np.random.random((1000,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e2), estimator='Natural')
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."


def test_TPCF_randoms():
    """
    test the possible randoms + PBCs combinations
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    #No PBCs w/ randoms
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Natural')
    #PBCs w/o randoms
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=None, period = period, 
                    max_sample_size=int(1e4), estimator='Natural')
    #PBCs w/ randoms
    result_3 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural')
    
    #No PBCs and no randoms should throw an error.
    try:
        tpcf(sample1, rbins, sample2 = sample2, 
             randoms=None, period = None, 
             max_sample_size=int(1e4), estimator='Natural')
    except HalotoolsError:
        pass
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_3)==3, "wrong number of correlation functions returned erroneously."


def test_TPCF_period_API():
    """
    test the period API functionality.
    """
    
    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    
    result_1 = tpcf(sample1, rbins, sample2 = sample2,
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural')
    
    period = 1.0
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural')
    
    #should throw an error.  period must be positive!
    period = np.array([1.0,1.0,-1.0])
    try:
        tpcf(sample1, rbins, sample2 = sample2, 
             randoms=randoms, period = period, 
             max_sample_size=int(1e4), estimator='Natural')
    except HalotoolsError:
        pass
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of correlation functions returned erroneously."


