# -*- coding: utf-8 -*-

"""
helper functions for clustering statistics functions, e.g. two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['_tpcf_process_args', '_tpcf_jackknife_process_args',\
           '_redshift_space_tpcf_process_args', '_s_mu_tpcf_process_args',\
           '_marked_tpcf_process_args','_delta_sigma_process_args',\
           '_list_estimators', '_TP_estimator', '_TP_estimator_requirements']

import numpy as np
from warnings import warn
from multiprocessing import cpu_count 

from ..custom_exceptions import *
from ..utils.array_utils import convert_to_ndarray, array_is_monotonic



def _tpcf_process_args(sample1, rbins, sample2, randoms, 
    period, do_auto, do_cross, estimator, num_threads, max_sample_size,
    approx_cell1_size, approx_cell2_size, approx_cellran_size):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.tpcf`. 
    """

    sample1 = convert_to_ndarray(sample1)

    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)

        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("Warning: sample1 and sample2 are exactly the same, \n"
                   "auto-correlation will be returned.\n")
            warn(msg)
            do_cross==False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        _sample1_is_sample2 = True

    if randoms is not None: 
        randoms = convert_to_ndarray(randoms)
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('downsampling sample1...')
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('down sampling sample1...')
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            print('down sampling sample2...')
    
    rbins = convert_to_ndarray(rbins)
    rmax = np.max(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rbins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    #check for input parameter consistency
    if (period is not None):
        if (rmax >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)

    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\nSample1 and sample2 must have same dimension.\n')
        raise HalotoolsError(msg)

    if (randoms is None) & (PBCs == False):
        msg = ('\nIf no PBCs are specified, randoms must be provided.\n')
        raise HalotoolsError(msg)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('do_auto and do_cross keywords must be of type boolean.')
        raise HalotoolsError(msg)

    if num_threads == 'max':
        num_threads = cpu_count()

    available_estimators = _list_estimators()
    if estimator not in available_estimators:
        msg = ("Input `estimator` must be one of the following:{0}".value(available_estimators))
        raise HalotoolsError(msg)


    return sample1, rbins, sample2, randoms, period, do_auto, do_cross, num_threads,\
           _sample1_is_sample2, PBCs


def _tpcf_jackknife_process_args(sample1, randoms, rbins, Nsub, sample2, period, do_auto,\
                                 do_cross, estimator, num_threads, max_sample_size):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.jackknife_tpcf`. 
    """

    sample1 = convert_to_ndarray(sample1)

    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)

        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("Warning: sample1 and sample2 are exactly the same, \n"
                   "auto-correlation will be returned.\n")
            warn(msg)
            do_cross==False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        _sample1_is_sample2 = True

    if randoms is not None: 
        randoms = convert_to_ndarray(randoms)
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('downsampling sample1...')
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('down sampling sample1...')
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            print('down sampling sample2...')
    
    rbins = convert_to_ndarray(rbins)
    rmax = np.max(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rbins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
    
    #Process Nsub entry and check for consistency.
    Nsub = convert_to_ndarray(Nsub)
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0]]*3)
    try:
        assert np.all(Nsub < np.inf)
        assert np.all(Nsub > 0)
    except AssertionError:
        msg = "Input ``Nsub`` must be a bounded positive number in all dimensions"
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    #check for input parameter consistency
    if (period is not None):
        if (rmax >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)

    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\nSample1 and sample2 must have same dimension.\n')
        raise HalotoolsError(msg)

    if (randoms is None) & (PBCs == False):
        msg = ('\nIf no PBCs are specified, randoms must be provided.\n')
        raise HalotoolsError(msg)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('do_auto and do_cross keywords must be of type boolean.')
        raise HalotoolsError(msg)

    if num_threads == 'max':
        num_threads = cpu_count()

    available_estimators = _list_estimators()
    if estimator not in available_estimators:
        msg = ("Input `estimator` must be one of the following:{0}".value(available_estimators))
        raise HalotoolsError(msg)


    return sample1, rbins, Nsub, sample2, randoms, period, do_auto, do_cross,\
           num_threads, _sample1_is_sample2, PBCs


def _redshift_space_tpcf_process_args(sample1, rp_bins, pi_bins, sample2, randoms,\
                                      period, do_auto, do_cross, estimator,\
                                      num_threads, max_sample_size=int(1e6)):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.redshift_space_tpcf`. 
    """

    sample1 = convert_to_ndarray(sample1)

    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)

        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("Warning: sample1 and sample2 are exactly the same, \n"
                   "auto-correlation will be returned.\n")
            warn(msg)
            do_cross==False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        _sample1_is_sample2 = True

    if randoms is not None: 
        randoms = convert_to_ndarray(randoms)
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('downsampling sample1...')
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('down sampling sample1...')
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            print('down sampling sample2...')
    
    #process projected radial bins
    rp_bins = convert_to_ndarray(rp_bins)
    rp_max = np.max(rp_bins)
    try:
        assert rp_bins.ndim == 1
        assert len(rp_bins) > 1
        if len(rp_bins) > 2:
            assert array_is_monotonic(rp_bins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rp_bins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
    
    #process parallel radial bins
    pi_bins = convert_to_ndarray(pi_bins)
    pi_max = np.max(pi_bins)
    try:
        assert pi_bins.ndim == 1
        assert len(pi_bins) > 1
        if len(pi_bins) > 2:
            assert array_is_monotonic(pi_bins, strict = True) == 1
    except AssertionError:
        msg = "Input ``pi_bins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    #check for input parameter consistency
    if (period is not None):
        if (rp_max >= np.min(period[0:2])/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)
    if (period is not None):
        if (pi_max >= np.min(period[2])/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)

    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\nSample1 and sample2 must have same dimension.\n')
        raise HalotoolsError(msg)

    if (randoms is None) & (PBCs == False):
        msg = ('\nIf no PBCs are specified, randoms must be provided.\n')
        raise HalotoolsError(msg)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('do_auto and do_cross keywords must be of type boolean.')
        raise HalotoolsError(msg)

    if num_threads == 'max':
        num_threads = cpu_count()

    available_estimators = _list_estimators()
    if estimator not in available_estimators:
        msg = ("Input `estimator` must be one of the following:{0}".value(available_estimators))
        raise HalotoolsError(msg)


    return sample1, rp_bins, pi_bins, sample2, randoms, period, do_auto, do_cross, num_threads, _sample1_is_sample2, PBCs


def _s_mu_tpcf_process_args(sample1, s_bins, mu_bins, sample2, randoms,\
                            period, do_auto, do_cross, estimator,\
                            num_threads, max_sample_size=int(1e6)):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.s_mu_tpcf`. 
    """

    sample1 = convert_to_ndarray(sample1)

    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)

        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("Warning: sample1 and sample2 are exactly the same, \n"
                   "auto-correlation will be returned.\n")
            warn(msg)
            do_cross==False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        _sample1_is_sample2 = True

    if randoms is not None: 
        randoms = convert_to_ndarray(randoms)
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('downsampling sample1...')
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            print('down sampling sample1...')
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            print('down sampling sample2...')
    
    #process radial bins
    s_bins = convert_to_ndarray(s_bins)
    s_max = np.max(s_bins)
    try:
        assert s_bins.ndim == 1
        assert len(s_bins) > 1
        if len(s_bins) > 2:
            assert array_is_monotonic(s_bins, strict = True) == 1
    except AssertionError:
        msg = "Input ``s_bins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
    
    #process angular bins
    mu_bins = convert_to_ndarray(mu_bins)
    
    #work with the sine of the angle between s and the LOS.  Only using cosine as the 
    #input because of convention.  sin(theta_los) increases as theta_los increases, which
    #is required in order to get the pair counter to work.  see note in cpairs s_mu_pairs.
    theta = np.arccos(mu_bins)
    mu_bins = np.sin(theta)[::-1] #must be increasing, remember to reverse result.
    
    mu_max = np.max(mu_bins)
    try:
        assert mu_bins.ndim == 1
        assert len(mu_bins) > 1
        if len(mu_bins) > 2:
            assert array_is_monotonic(mu_bins, strict = True) == 1
    except AssertionError:
        msg = "Input ``mu_bins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
    
    if (np.min(mu_bins)<0.0) | (np.max(mu_bins)>1.0):
        raise ValueError('mu bins must be in the range [0,1]')
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    #check for input parameter consistency
    if (period is not None):
        if (s_max >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)
    
    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\nSample1 and sample2 must have same dimension.\n')
        raise HalotoolsError(msg)

    if (randoms is None) & (PBCs == False):
        msg = ('\nIf no PBCs are specified, randoms must be provided.\n')
        raise HalotoolsError(msg)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('do_auto and do_cross keywords must be of type boolean.')
        raise HalotoolsError(msg)

    if num_threads == 'max':
        num_threads = cpu_count()

    available_estimators = _list_estimators()
    if estimator not in available_estimators:
        msg = ("Input `estimator` must be one of the following:{0}".value(available_estimators))
        raise HalotoolsError(msg)


    return sample1, s_bins, mu_bins, sample2, randoms, period, do_auto, do_cross, num_threads, _sample1_is_sample2, PBCs


def _marked_tpcf_process_args(sample1, rbins, sample2, marks1, marks2,\
                              period, do_auto, do_cross, num_threads,\
                              max_sample_size, wfunc):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.mared_tpcf`. 
    """

    sample1 = convert_to_ndarray(sample1)

    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)

        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("Warning: sample1 and sample2 are exactly the same, \n"
                   "auto-correlation will be returned.\n")
            warn(msg)
            do_cross==False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        _sample1_is_sample2 = True

    #process wfunc parameter
    if type(wfunc) is not int:
        raise ValueError('wfunc parameter must be an integer ID of the desired function.')

    #process marks
    if marks1 is not None: 
        marks1 = convert_to_ndarray(marks1).astype(float)
    else:
        marks1 = np.ones(len(sample1)).astype(float)
    if marks2 is not None: 
        marks2 = convert_to_ndarray(marks2).astype(float)
    else:
        marks2 = np.ones(len(sample2)).astype(float)
    
    #check for consistency between marks and samples
    if len(marks1) != len(sample1):
        msg = ("marks1 must have same length as sample1")
        warn(msg)
    if len(marks2) != len(marks2):
        msg = ("marks2 must have same length as sample2")
        warn(msg)
    if marks1.ndim != 1:
        msg = "marks1 must be a one dimensional array"
        raise HalotoolsError(msg)
    if marks2.ndim != 1:
        msg = "marks2 must be a one dimensional array"
        raise HalotoolsError(msg)
        
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            marks1 = marks1[inds]
            print('downsampling sample1...')
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            marks1 = marks1[inds]
            print('down sampling sample1...')
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            marks2 = marks2[inds]
            print('down sampling sample2...')
    
    rbins = convert_to_ndarray(rbins)
    rmax = np.max(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rbins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    #check for input parameter consistency
    if (period is not None):
        if (rmax >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)

    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\nSample1 and sample2 must have same dimension.\n')
        raise HalotoolsError(msg)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('do_auto and do_cross keywords must be of type boolean.')
        raise HalotoolsError(msg)

    if num_threads == 'max':
        num_threads = cpu_count()

    return sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross,\
           num_threads, wfunc, _sample1_is_sample2, PBCs


def _delta_sigma_process_args(galaxies, particles, rp_bins, chi_max, period,\
                              estimator, num_threads):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.delta_sigma`. 
    """

    galaxies = convert_to_ndarray(galaxies)
    particles = convert_to_ndarray(particles)

    rp_bins = convert_to_ndarray(rp_bins)
    rp_max = np.max(rp_bins)
    try:
        assert rp_bins.ndim == 1
        assert len(rp_bins) > 1
        if len(rp_bins) > 2:
            assert array_is_monotonic(rp_bins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rp_bins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    if num_threads == 'max':
        num_threads = cpu_count()

    available_estimators = _list_estimators()
    if estimator not in available_estimators:
        msg = ("Input `estimator` must be one of the following:{0}".value(available_estimators))
        raise HalotoolsError(msg)


    return galaxies, particles, rp_bins, period, num_threads, PBCs


def _list_estimators():
    """
    private internal function.
    
    list available tpcf estimators
    """
    estimators = ['Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay']
    return estimators


def _TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
    """
    private internal function.
    
    two point correlation function estimator
    
    note: This buisness of using np.outer is to make this work for the jackknife function
    which returns an numpy.ndarray.
    """
    
    ND1 = convert_to_ndarray(ND1)
    ND2 = convert_to_ndarray(ND2)
    NR1 = convert_to_ndarray(NR1)
    NR2 = convert_to_ndarray(NR2)
    Ns = np.array([len(ND1),len(ND2),len(NR1),len(NR2)])
    
    if np.any(Ns>1):
        mult = np.outer #used for the jackknife calculations
    else:
        mult = lambda x,y: x*y #used for all else
    
    if estimator == 'Natural':
        factor = ND1*ND2/(NR1*NR2)
        #DD/RR-1
        xi = mult(1.0/factor,DD/RR) - 1.0
    elif estimator == 'Davis-Peebles':
        factor = ND1*ND2/(ND1*NR2)
        #DD/DR-1
        xi = mult(1.0/factor,DD/DR) - 1.0
    elif estimator == 'Hewett':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD-DR)/RR
        xi = mult(1.0/factor1,DD/RR) - mult(1.0/factor2,DR/RR)
    elif estimator == 'Hamilton':
        #DDRR/DRDR-1
        xi = (DD*RR)/(DR*DR) - 1.0
    elif estimator == 'Landy-Szalay':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD - 2.0*DR + RR)/RR
        xi = mult(1.0/factor1,DD/RR) - mult(1.0/factor2,2.0*DR/RR) + 1.0
    else: 
        raise ValueError("unsupported estimator!")
    
    if np.shape(xi)[0]==1: return xi[0]
    else: return xi


def _TP_estimator_requirements(estimator):
    """
    private internal function.
    
    return booleans indicating which pairs need to be counted for the chosen estimator
    """
    if estimator == 'Natural':
        do_DD = True
        do_DR = False
        do_RR = True
    elif estimator == 'Davis-Peebles':
        do_DD = True
        do_DR = True
        do_RR = False
    elif estimator == 'Hewett':
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == 'Hamilton':
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == 'Landy-Szalay':
        do_DD = True
        do_DR = True
        do_RR = True
    else: 
        available_estimators = _list_estimators()
        if estimator not in available_estimators:
            msg = ("Input `estimator` must be one of the following:{0}".value(available_estimators))
            raise HalotoolsError(msg)

    return do_DD, do_DR, do_RR



