#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..tpcf import tpcf
from ...custom_exceptions import *

import pytest
slow = pytest.mark.slow

__all__=['test_tpcf_auto', 'test_tpcf_cross', 'test_tpcf_estimators',\
         'test_tpcf_sample_size_limit',\
         'test_tpcf_randoms', 'test_tpcf_period_API']

"""
Note that these are almost all unit-tests.  Non tirival tests are a little heard to think
of here.
"""

@slow
def test_tpcf_auto():
    """
    test the tpcf auto-correlation functionality
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #with randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural', 
                  approx_cell1_size = [rmax, rmax, rmax], 
                  approx_cellran_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."
    
    #with out randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=None, period = period, 
                  max_sample_size=int(1e4), estimator='Natural', 
                  approx_cell1_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."


@slow
def test_tpcf_cross():
    """
    test the tpcf cross-correlation functionality
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #with randoms
    result = tpcf(sample1, rbins, sample2 = sample2, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural', do_auto=False, 
                  approx_cell1_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."
    
    #with out randoms
    result = tpcf(sample1, rbins, sample2 = sample2, 
                  randoms=None, period = period, 
                  max_sample_size=int(1e4), estimator='Natural', do_auto=False, 
                  approx_cell1_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."

@slow
def test_tpcf_estimators():
    """
    test the tpcf different estimators functionality
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Davis-Peebles', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_3 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Hewett', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_4 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Hamilton', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_5 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Landy-Szalay', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
                                            
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of  correlation functions returned erroneously."
    assert len(result_3)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_4)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_5)==3, "wrong number of correlation functions returned erroneously."

@slow
def test_tpcf_sample_size_limit():
    """
    test the tpcf sample size limit functionality functionality
    """
    
    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((1000,3))
    randoms = np.random.random((1000,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e2), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax])
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."

@slow
def test_tpcf_randoms():
    """
    test the tpcf possible randoms + PBCs combinations
    """
    
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #No PBCs w/ randoms
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    #PBCs w/o randoms
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=None, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    #PBCs w/ randoms
    result_3 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    
    #No PBCs and no randoms should throw an error.
    try:
        tpcf(sample1, rbins, sample2 = sample2, 
             randoms=None, period = None, 
             max_sample_size=int(1e4), estimator='Natural', 
             approx_cell1_size = [rmax, rmax, rmax], 
             approx_cellran_size = [rmax, rmax, rmax])
    except HalotoolsError:
        pass
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_3)==3, "wrong number of correlation functions returned erroneously."

@slow
def test_tpcf_period_API():
    """
    test the tpcf period API functionality.
    """
    
    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()
    
    result_1 = tpcf(sample1, rbins, sample2 = sample2,
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax])
    
    period = 1.0
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax])
    
    #should throw an error.  period must be positive!
    period = np.array([1.0,1.0,-1.0])
    try:
        tpcf(sample1, rbins, sample2 = sample2, 
             randoms=randoms, period = period, 
             max_sample_size=int(1e4), estimator='Natural', 
             approx_cell1_size = [rmax, rmax, rmax])
    except HalotoolsError:
        pass
    
    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of correlation functions returned erroneously."


