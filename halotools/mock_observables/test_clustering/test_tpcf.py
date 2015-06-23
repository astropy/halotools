#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import sys
from ..clustering import tpcf

__all__=['test_TPCF_auto', 'test_TPCF_estimator', 'test_TPCF_sample_size_limit',\
         'test_TPCF_randoms', 'test_TPCF_period_API']

####two point correlation function########################################################

def test_TPCF_auto():
    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
    #with randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural')
    assert result.ndim == 1, "More than one correlation function returned erroneously."

def test_TPCF_estimator():

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
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
                                            
    
    assert len(result_1)==3, "One or more correlation functions returned erroneously."
    assert len(result_2)==3, "One or more correlation functions returned erroneously."
    assert len(result_3)==3, "One or more correlation functions returned erroneously."
    assert len(result_4)==3, "One or more correlation functions returned erroneously."
    assert len(result_5)==3, "One or more correlation functions returned erroneously."


def test_TPCF_sample_size_limit():

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                                              randoms=randoms, period = None, 
                                              max_sample_size=int(1e2), estimator='Natural')
    
    assert len(result_1)==3, "One or more correlation functions returned erroneously."


def test_TPCF_randoms():

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.4,5)
    
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
    
    #know how to make sure this throws a valueerror?
    '''
    #no randoms, no PBCs, should throw error!
    two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=None, period = None, 
                                            max_sample_size=int(1e4), estimator='Natural')
    '''
    
    assert len(result_1)==3, "One or more correlation functions returned erroneously."
    assert len(result_2)==3, "One or more correlation functions returned erroneously."
    assert len(result_3)==3, "One or more correlation functions returned erroneously."


def test_TPCF_period_API():

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.4,5)
    
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = period, 
                                            max_sample_size=int(1e4), estimator='Natural')
    period = 1
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = period, 
                                            max_sample_size=int(1e4), estimator='Natural')
    period = np.array([1.0,1.0,np.inf])
    '''
    #should throw error!
    result_2 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = period, 
                                            max_sample_size=int(1e4), estimator='Natural')
    '''
                                            
    
    assert len(result_1)==3, "One or more correlation functions returned erroneously."
    assert len(result_2)==3, "One or more correlation functions returned erroneously."
##########################################################################################