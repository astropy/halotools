# -*- coding: utf-8 -*-

"""
functions to calculate clustering statistics, e.g. two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['_list_estimators', '_TP_estimator', '_TP_estimator_requirements']

import numpy as np
from warnings import warn

from ..custom_exceptions import *
from ..utils.array_utils import convert_to_ndarray, array_is_monotonic



def tpcf_process_args(sample1, rbins, sample2=None, randoms=None, period=None,
    do_auto=True, do_cross=True, estimator='Natural', N_threads=1,
    max_sample_size=int(1e6)):


    #process input parameters
    sample1 = convert_to_ndarray(sample1)

    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)

        if np.all(sample1==sample2):
            _sample1_equals_sample2 = True
            msg = ("Warning: sample1 and sample2 are exactly the same, \n"
                   "auto-correlation will be returned.\n")
            warn(msg)
            do_cross==False
    else: 
        sample2 = sample1

    if randoms is not None: 
        randoms = convert_to_ndarray(randoms)

    rbins = convert_to_ndarray(rbins)
    
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
    xperiod, yperiod, zperiod = period 
    
    #down sample if sample size exceeds max_sample_size.
    if (len(sample1)>max_sample_size) & (np.all(sample1==sample2)):
        inds = np.arange(0,len(sample1))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = sample1[inds]
        sample2 = sample2[inds]
        print('downsampling sample1...')
    if len(sample2)>max_sample_size:
        inds = np.arange(0,len(sample2))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample2 = sample2[inds]
        print('down sampling sample2...')
    if len(sample1)>max_sample_size:
        inds = np.arange(0,len(sample1))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = sample1[inds]
        print('down sampling sample1...')
    
    #check radial bins
    if np.shape(rbins) == ():
        rbins = np.array([rbins])
    if rbins.ndim != 1:
        raise ValueError('rbins must be a 1-D array')
    if len(rbins)<2:
        raise ValueError('rbins must be of lenght >=2.')
    
    #check dimensionality of data. currently, points must be 3D.
    k = np.shape(sample1)[-1]
    if k!=3:
        raise ValueError('data must be 3-dimensional.')
    
    #check for input parameter consistency
    if (period is not None) & (np.max(rbins)>np.min(period)/2.0):
        raise ValueError('cannot calculate for seperations larger than Lbox/2.')
    if (sample2 is not None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('sample1 and sample2 must have same dimension.')
    if (randoms is None) & (min(period)==np.inf):
        raise ValueError('if no PBCs are specified, randoms must be provided.')
    if estimator not in estimators: 
        raise ValueError('user must specify a supported estimator. Supported estimators \
        are:{0}'.value(estimators))
    if (PBCs==True) & (max(period)==np.inf):
        raise ValueError('if a non-infinte PBC specified, all PBCs must be non-infinte.')
    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        raise ValueError('do_auto and do_cross keywords must be of type boolean.')



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
        raise ValueError("unsupported estimator!")
    return do_DD, do_DR, do_RR



