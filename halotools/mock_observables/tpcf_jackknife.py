# -*- coding: utf-8 -*-

"""
functions to calculate clustering statistics, e.g. two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma

from .clustering_helpers import *
from .pair_counters.rect_cuboid_pairs import npairs, xy_z_npairs, jnpairs, s_mu_npairs
from .pair_counters.double_tree_pairs import npairs
##########################################################################################


__all__=['tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def tpcf_jackknife(sample1, randoms, rbins, Nsub=[5,5,5], Lbox=[250.0,250.0,250.0],\
                   sample2=None, period=None, do_auto=True, do_cross=True,\
                   estimator='Natural', num_threads=1, max_sample_size=int(1e6)):
    """
    Calculate the two-point correlation function, :math:`\\xi(r)` and the covariance 
    matrix.
    
    The covariance matrix is calculated using spatial jackknife sampling of the simulation 
    box.  The spatial samples are defined by splitting the box along each dimension set by
    the `Nsub` argument.
    
    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    randoms : array_like
        Nran x 3 numpy array containing 3-D positions of points.
    
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    Nsub : array_like, optional
        numpy array of number of divisions along each dimension defining jackknife 
        subvolumes.  If single integer is given, assumed to be equivalent for each 
        dimension.  Total number of jackknife samples is numpy.prod(`Nsub`).
    
    Lbox : array_like, optional
        length of data volume along each dimension.
    
    sample2 : array_like, optional
        Npts x 3 numpy array containing 3-D positions of points.
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be numpy.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    do_auto : boolean, optional
        do auto-correlation?  Default is True.
    
    do_cross : boolean, optional
        do cross-correlation?  Default is True.
    
    estimator : string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsample is equal to max_sample_size. 

    Returns 
    -------
    correlation_function(s), cov_matrix(ices) : numpy.array, numpy.ndarray
        len(`rbins`)-1 length array containing correlation function :math:`\\xi(r)` 
        computed in each of the radial bins defined by input `rbins`.
        
        len(rbins)-1 x len(rbins)-1 ndarray containing the covariance matrix of `\\xi(r)`
        
        If `sample2` is passed as input, three arrays of length len(`rbins`)-1 are 
        returned: :math:`\\xi_{11}(r)`, :math:`\\xi_{12}(r)`, :math:`\\xi_{22}(r)`,
        and three arrays of shape len(`rbins`)-1 by len(`rbins`)-1
        :math: `\\mathrm{cov}_{11}`, :math:`\\mathrm{cov}_{12}`, 
        :math:`\\mathrm{cov}_{22}`, are returned.
        
        The autocorrelation of `sample1`, the cross-correlation between `sample1` and 
        `sample2`, and the autocorrelation of `sample2`, and the associated covariance 
        matrices.  If `do_auto` or `do_cross` is set to False, the appropriate result(s) 
        is not returned.
    
    Notes
    -----
    The jackknife sampling of pair counts is done internally to the pair counter.  Pairs 
    are counted for each jackknife sample such that if both pairs are in the current 
    sample, they contribute +1 count, if one pair is inside, and one outside, +0.5 
    counts, and if both are outside, +0 counts.
    """
    
    estimators = _list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if sample2 != None: 
        sample2 = np.asarray(sample2)
        if np.all(sample1==sample2):
            do_cross==False
            print("Warning: sample1 and sample2 are exactly the same, only the\
                   auto-correlation will be returned.")
    else: sample2 = sample1
    randoms = np.asarray(randoms)
    rbins = np.asarray(rbins)
    if type(Nsub) is int: Nsub = np.array([Nsub]*np.shape(sample1)[-1])
    else: Nsub = np.asarray(Nsub)
    if type(Lbox) in (int,float): Lbox = np.array([Lbox]*np.shape(sample1)[-1])
    else: Lbox = np.asarray(Lbox)
    #Process period entry and check for consistency.
    if period is None:
            PBCs = False
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        PBCs = True
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have shape (k,)")
            return None
    #down sample is sample size exceeds max_sample_size.
    if (len(sample2)>max_sample_size) & (not np.all(sample1==sample2)):
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
    if len(randoms)>max_sample_size:
        inds = np.arange(0,len(randoms))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = randoms[inds]
        print('down sampling randoms...')
    if np.shape(Nsub)[0]!=np.shape(sample1)[-1]:
        raise ValueError("Nsub should have shape (k,) or be a single integer")
    
    #check radial bins
    if np.shape(rbins) == ():
        rbins = np.array([rbins])
    if rbins.ndim != 1:
        raise ValueError('rbins must be a 1-D array')
    if len(rbins)<2:
        raise ValueError('rbins must be of lenght >=2.')
    
    k = np.shape(sample1)[-1] #dimensionality of data
    if k!=3:
        raise ValueError('data must be 3-dimensional.')
        
    N1 = len(sample1)
    N2 = len(sample2)
    Nran = len(randoms)
    
    #check for input parameter consistency
    if (period is not None) & (np.max(rbins)>np.min(period)/2.0):
        raise ValueError('Cannot calculate for seperations larger than Lbox/2.')
    if (sample2 is not None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('Sample 1 and sample 2 must have same dimension.')
    if estimator not in estimators: 
        raise ValueError('Must specify a supported estimator. Supported estimators are:{0}'
        .value(estimators))
    if (PBCs==True) & (max(period)==np.inf):
        raise ValueError('If a non-infinte PBC specified, all PBCs must be non-infinte.')
    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        raise ValueError('do_auto and do_cross keywords must be of type boolean.')
    
    def get_subvolume_labels(sample1, sample2, randoms, Nsub, Lbox):
        """
        Split volume into subvolumes, and tag points in subvolumes with integer labels for 
        use in the jackknife calculation.
        
        note: '0' tag should be reserved. It is used in the jackknife calculation to mean
        'full sample'
        """
        
        dL = Lbox/Nsub # length of subvolumes along each dimension
        N_sub_vol = np.prod(Nsub) # total the number of subvolumes
        inds = np.arange(1,N_sub_vol+1).reshape(Nsub[0],Nsub[1],Nsub[2])
    
        #tag each particle with an integer indicating which jackknife subvolume it is in
        #subvolume indices for the sample1 particle's positions
        index_1 = np.floor(sample1/dL).astype(int)
        j_index_1 = inds[index_1[:,0],index_1[:,1],index_1[:,2]].astype(int)
    
        #subvolume indices for the random particle's positions
        index_random = np.floor(randoms/dL).astype(int)
        j_index_random = inds[index_random[:,0],\
                              index_random[:,1],\
                              index_random[:,2]].astype(int)
        
        #subvolume indices for the sample2 particle's positions
        index_2 = np.floor(sample2/dL).astype(int)
        j_index_2 = inds[index_2[:,0],index_2[:,1],index_2[:,2]].astype(int)
        
        return j_index_1, j_index_2, j_index_random, int(N_sub_vol)
    
    def get_subvolume_numbers(j_index,N_sub_vol):
        #need every label to be in there at least once
        temp = np.hstack((j_index,np.arange(1,N_sub_vol+1,1)))
        labels, N = np.unique(temp,return_counts=True)
        N = N-1 #remove the place holder I added two lines above.
        return N
    
    def jnpair_counts(sample1, sample2, j_index_1, j_index_2, N_sub_vol, rbins,\
                      period, N_thread, do_auto, do_cross, do_DD):
        """
        Count jackknife data pairs: DD
        """
        if do_auto==True:
            D1D1 = jnpairs(sample1, sample1, rbins, period=period,\
                           jtags1=j_index_1, jtags2=j_index_1,  N_samples=N_sub_vol,\
                           num_threads=num_threads)
            D1D1 = np.diff(D1D1,axis=1)
        else:
            D1D1=None
            D2D2=None
        
        if np.all(sample1 == sample2):
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = jnpairs(sample1, sample2, rbins, period=period,\
                               jtags1=j_index_1, jtags2=j_index_2,\
                               N_samples=N_sub_vol, num_threads=num_threads)
                D1D2 = np.diff(D1D2,axis=1)
            else: D1D2=None
            if do_auto==True:
                D2D2 = jnpairs(sample2, sample2, rbins, period=period,\
                               jtags1=j_index_2, jtags2=j_index_2,\
                               N_samples=N_sub_vol, num_threads=num_threads)
                D2D2 = np.diff(D2D2,axis=1)

        return D1D1, D1D2, D2D2
    
    def jrandom_counts(sample, randoms, j_index, j_index_randoms, N_sub_vol, rbins,\
                       period, N_thread, do_DR, do_RR):
        """
        Count jackknife random pairs: DR, RR
        """
        
        if do_DR==True:
            DR = jnpairs(sample, randoms, rbins, period=period,\
                         jtags1=j_index, jtags2=j_index_randoms,\
                          N_samples=N_sub_vol, num_threads=num_threads)
            DR = np.diff(DR,axis=1)
        else: DR=None
        if do_RR==True:
            RR = jnpairs(randoms, randoms, rbins, period=period,\
                         jtags1=j_index_randoms, jtags2=j_index_randoms,\
                         N_samples=N_sub_vol, num_threads=num_threads)
            RR = np.diff(RR,axis=1)
        else: RR=None

        return DR, RR
    
    def jackknife_errors(sub,full,N_sub_vol):
        """
        Calculate jackknife errors.
        """
        after_subtraction =  sub - np.mean(sub,axis=0)
        squared = after_subtraction**2.0
        error2 = ((N_sub_vol-1)/N_sub_vol)*squared.sum(axis=0)
        error = error2**0.5
        
        return error
    
    def covariance_matrix(sub,full,N_sub_vol):
        """
        Calculate the full covariance matrix.
        """
        Nr = full.shape[0] # Nr is the number of radial bins
        cov = np.zeros((Nr,Nr)) # 2D array that keeps the covariance matrix 
        after_subtraction = sub - np.mean(sub,axis=0)
        tmp = 0
        for i in range(Nr):
            for j in range(Nr):
                tmp = 0.0
                for k in range(N_sub_vol):
                    tmp = tmp + after_subtraction[k,i]*after_subtraction[k,j]
                cov[i,j] = (((N_sub_vol-1)/N_sub_vol)*tmp)
    
        return cov
    
    def TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
        """
        two point correlation function estimator
        """
        if estimator == 'Natural':
            factor = ND1*ND2/(NR1*NR2)
            #DD/RR-1
            xi = (1.0/factor)*(DD/RR).T - 1.0
        elif estimator == 'Davis-Peebles':
            factor = ND1*ND2/(ND1*NR2)
            #DD/DR-1
            xi = (1.0/factor)*(DD/DR).T - 1.0
        elif estimator == 'Hewett':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            #(DD-DR)/RR
            xi = (1.0/factor1)*(DD/RR).T - (1.0/factor2)*(DR/RR).T
        elif estimator == 'Hamilton':
            #DDRR/DRDR-1
            xi = (DD*RR)/(DR*DR) - 1.0
        elif estimator == 'Landy-Szalay':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            #(DD - 2.0*DR + RR)/RR
            xi = (1.0/factor1)*(DD/RR).T - (1.0/factor2)*2.0*(DR/RR).T + 1.0
        else: 
            raise ValueError("unsupported estimator!")
        return xi.T #transpose 2 times to get the multiplication to work
    
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
    
    N1 = len(sample1)
    N2 = len(sample2)
    NR = len(randoms)
    
    j_index_1, j_index_2, j_index_random, N_sub_vol = \
                               get_subvolume_labels(sample1, sample2, randoms, Nsub, Lbox)
    
    #number of points in each subvolume
    NR_subs = get_subvolume_numbers(j_index_random,N_sub_vol)
    N1_subs = get_subvolume_numbers(j_index_1,N_sub_vol)
    N2_subs = get_subvolume_numbers(j_index_2,N_sub_vol)
    #number of points in each jackknife sample
    N1_subs = N1 - N1_subs
    N2_subs = N2 - N2_subs
    NR_subs = NR - NR_subs
    
    #calculate all the pair counts
    D1D1, D1D2, D2D2 = jnpair_counts(sample1, sample2, j_index_1, j_index_2, N_sub_vol,\
                                     rbins, period, num_threads, do_auto, do_cross, do_DD)
    D1D1_full = D1D1[0,:]
    D1D1_sub = D1D1[1:,:]
    D1D2_full = D1D2[0,:]
    D1D2_sub = D1D2[1:,:]
    D2D2_full = D2D2[0,:]
    D2D2_sub = D2D2[1:,:]
    D1R, RR = jrandom_counts(sample1, randoms, j_index_1, j_index_random, N_sub_vol,\
                             rbins, period, num_threads, do_DR, do_RR)
    if np.all(sample1==sample2):
        D2R=D1R
    else:
        if do_DR==True:
            D2R, RR_dummy= jrandom_counts(sample2, randoms, j_index_2, j_index_random,\
                                          N_sub_vol, rbins, period, num_threads, do_DR,
                                          do_RR=False)
        else: D2R = None
    
    if do_DR==True:    
        D1R_full = D1R[0,:]
        D1R_sub = D1R[1:,:]
        D2R_full = D2R[0,:]
        D2R_sub = D2R[1:,:]
    else:
        D1R_full = None
        D1R_sub = None
        D2R_full = None
        D2R_sub = None
    if do_RR==True:
        RR_full = RR[0,:]
        RR_sub = RR[1:,:]
    else:
        RR_full = None
        RR_sub = None
    
    
    #calculate the correlation function for the full sample
    xi_11_full = TP_estimator(D1D1_full, D1R_full, RR_full, N1, N1, NR, NR, estimator)
    xi_12_full = TP_estimator(D1D2_full, D1R_full, RR_full, N1, N2, NR, NR, estimator)
    xi_22_full = TP_estimator(D2D2_full, D2R_full, RR_full, N2, N2, NR, NR, estimator)
    #calculate the correlation function for the subsamples
    xi_11_sub = TP_estimator(D1D1_sub, D1R_sub, RR_sub, N1_subs, N1_subs, NR_subs,\
                             NR_subs, estimator)
    xi_12_sub = TP_estimator(D1D2_sub, D1R_sub, RR_sub, N1_subs, N2_subs, NR_subs,\
                             NR_subs, estimator)
    xi_22_sub = TP_estimator(D2D2_sub, D2R_sub, RR_sub, N2_subs, N2_subs, NR_subs,\
                             NR_subs, estimator)
    
    #calculate the errors
    xi_11_err = jackknife_errors(xi_11_sub,xi_11_full,N_sub_vol)
    xi_12_err = jackknife_errors(xi_12_sub,xi_12_full,N_sub_vol)
    xi_22_err = jackknife_errors(xi_22_sub,xi_22_full,N_sub_vol)
    
    #calculate the covariance matrix
    xi_11_cov = covariance_matrix(xi_11_sub,xi_11_full,N_sub_vol)
    xi_12_cov = covariance_matrix(xi_12_sub,xi_12_full,N_sub_vol)
    xi_22_cov = covariance_matrix(xi_22_sub,xi_22_full,N_sub_vol)
    
    if np.all(sample1==sample2):
        return xi_11_full,xi_11_cov
    else:
        if (do_auto==True) & (do_cross==True):
            return xi_11_full,xi_12_full,xi_22_full,xi_11_cov,xi_12_cov,xi_22_cov
        elif do_auto==True:
            return xi_11_full,xi_22_full,xi_11_cov,xi_22_cov
        elif do_cross==True:
            return xi_12_full,xi_12_cov
