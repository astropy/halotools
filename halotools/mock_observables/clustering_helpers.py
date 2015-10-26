# -*- coding: utf-8 -*-

"""
helper functions for clustering statistics functions, e.g. two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['_tpcf_process_args', '_tpcf_jackknife_process_args', '_list_estimators', '_TP_estimator', '_TP_estimator_requirements']

import numpy as np
from warnings import warn
from multiprocessing import cpu_count 

from ..custom_exceptions import *
from ..utils.array_utils import convert_to_ndarray, array_is_monotonic



def _tpcf_process_args(sample1, rbins, sample2, randoms, 
    period, do_auto, do_cross, estimator, num_threads, max_sample_size):
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


def _redshift_space_tpcf_process_args(sample1, rbins, sample2, randoms, 
    period, do_auto, do_cross, estimator, num_threads, max_sample_size):
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


    return sample1, rbins, sample2, randoms, period, do_auto, do_cross, num_threads, _sample1_is_sample2, PBCs


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
    
    note: jackknife_tpcf uses its own intenral version, this is not totally ideal.
    """
    if estimator == 'Natural':
        factor = ND1*ND2/(NR1*NR2)
        #DD/RR-1
        xi = (1.0/factor)*DD/RR - 1.0
    elif estimator == 'Davis-Peebles':
        factor = ND1*ND2/(ND1*NR2)
        #DD/DR-1
        xi = (1.0/factor)*DD/DR - 1.0
    elif estimator == 'Hewett':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD-DR)/RR
        xi = (1.0/factor1)*DD/RR - (1.0/factor2)*DR/RR
    elif estimator == 'Hamilton':
        #DDRR/DRDR-1
        xi = (DD*RR)/(DR*DR) - 1.0
    elif estimator == 'Landy-Szalay':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD - 2.0*DR + RR)/RR
        xi = (1.0/factor1)*DD/RR - (1.0/factor2)*2.0*DR/RR + 1.0
    else: 
        raise ValueError("unsupported estimator!")
    return xi


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



