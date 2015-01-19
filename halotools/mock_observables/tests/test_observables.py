#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
from ..observables import two_point_correlation_function
from ..observables import angular_two_point_correlation_function
from ..observables import two_point_correlation_function_jackknife
from ..observables import isolatoion_criterion
from ..observables import Delta_Sigma
from ..spatial import geometry

####two point correlation function########################################################
def test_TPCF_auto():

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
    #with randoms
    result = two_point_correlation_function(sample1, rbins, sample2 = None, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e4), estimator='Natural')
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_TPCF_estimator():

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
    result_1 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e4), estimator='Natural')
    result_2 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e4), estimator='Davis-Peebles')
    result_3 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e4), estimator='Hewett')
    result_4 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e4), estimator='Hamilton')
    result_5 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
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
    
    result_1 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e2), estimator='Natural')
    
    assert len(result_1)==3, "One or more correlation functions returned erroneously."


def test_TPCF_randoms():

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
    #No PBCs w/ randoms
    result_1 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = None, 
                                            max_sample_size=int(1e4), estimator='Natural')
    #PBCs w/o randoms
    result_2 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=None, period = period, 
                                            max_sample_size=int(1e4), estimator='Natural')
    #PBCs w/ randoms
    result_3 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
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
    rbins = np.linspace(0,0.5,5)
    
    result_1 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
                                            randoms=randoms, period = period, 
                                            max_sample_size=int(1e4), estimator='Natural')
    period = 1
    result_2 = two_point_correlation_function(sample1, rbins, sample2 = sample2, 
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


####two point correlation jackknife function##############################################
def test_two_point_correlation_function_jackknife():
    
    sample1 = np.random.random((100,3))
    randoms = np.random.random((1000,3))
    period = np.array([1,1,1])
    Lbox = np.array([1,1,1])
    rbins = np.linspace(0.0,0.5,5)
    
    result_1,err = two_point_correlation_function_jackknife(sample1, randoms, rbins, Nsub=5, Lbox=Lbox, period = period, N_threads=1)
    result_2 = two_point_correlation_function(sample1, rbins,  randoms=randoms, period = period, N_threads=1)
    
    assert np.all(result_1==result_2), "correlation functions do not match"

def test_two_point_correlation_function_jackknife_threading():
    
    sample1 = np.random.random((100,3))
    randoms = np.random.random((1000,3))
    period = np.array([1,1,1])
    Lbox = np.array([1,1,1])
    rbins = np.linspace(0.0,0.5,5)
    
    result_1,err = two_point_correlation_function_jackknife(sample1, randoms, rbins, Nsub=5, Lbox=Lbox, period = period, N_threads=4)
    result_2 = two_point_correlation_function(sample1, rbins,  randoms=randoms, period = period, N_threads=1)
    
    assert np.all(result_1==result_2), "correlation functions do not match"
##########################################################################################


####two point angular correlation function################################################
def test_angular_TPCF_auto():

    import halotools.utils.spherical_geometry as sg

    N1=100
    N2=100
    Nran = 1000

    sample1 = sg.sample_spherical_surface(N1)
    sample2 = sg.sample_spherical_surface(N2)
    randoms = sg.sample_spherical_surface(Nran)
    
    theta_bins = np.linspace(0,180,10)
    
    #with randoms
    result = angular_two_point_correlation_function(sample1, theta_bins, sample2 = None, 
                                                    randoms=randoms, estimator='Natural')
    print(result)
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_angular_TPCF_cross():

    import halotools.utils.spherical_geometry as sg

    N1=100
    N2=100
    Nran = 1000

    sample1 = sg.sample_spherical_surface(N1)
    sample2 = sg.sample_spherical_surface(N2)
    randoms = sg.sample_spherical_surface(Nran)
    
    theta_bins = np.linspace(0,180,10)
    
    #with randoms
    result = angular_two_point_correlation_function(sample1, theta_bins, sample2 = sample2, 
                                                    randoms=randoms, estimator='Natural')
    result = np.asarray(result)
    
    assert np.shape(result)[0] == 3, "incorrect number/form of correlations returned"
##########################################################################################


####delta sigma function##################################################################
def test_delta_sigma():
    
    sample1 = np.random.random((10,3))
    sample2 = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0.1,0.5,4)
    
    result = Delta_Sigma(sample1, sample2, rbins, period=period)
    
    pass
##########################################################################################


####isolation criteria####################################################################
"""
def test_isolation_criterion_API():
    #define isolation function. This one works with magnitudes, to find galaxies with no 
    #neighbors brighter than host+0.5 mag
    def is_isolated(candidate_prop,other_prop):
        delta = 0.5
        return other_prop>(candidate_prop+delta)
    
    iso_crit = isolatoion_criterion(volume=geometry.sphere, test_func=is_isolated)
    
    from halotools import make_mocks
    mock = make_mocks.HOD_mock()
    mock.populate()
    
    result = iso_crit.apply_criterion(mock,[0])
    print(result)
    assert True==False
"""
##########################################################################################
    
    
    
    
    