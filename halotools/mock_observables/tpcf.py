# -*- coding: utf-8 -*-

"""
Calculate two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma
from .clustering_helpers import *
from .pair_counters.double_tree_pairs import npairs
##########################################################################################


__all__=['tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def tpcf(sample1, rbins, sample2=None, randoms=None, period=None,\
         do_auto=True, do_cross=True, estimator='Natural', num_threads=1,\
         max_sample_size=int(1e6)):
    """ 
    Calculate the real space two-point correlation function, :math:`\\xi(r)`.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are 
        counted.
    
    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.
    
    randoms : array_like, optional
        Npts x 3 array containing 3-D positions of points.  If no randoms are provided
        analytic randoms are used (only valid for periodic boundary conditions).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    do_auto : boolean, optional
        do auto-correlation?
    
    do_cross : boolean, optional
        do cross-correlation?
    
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
    correlation_function : numpy.array
        len(`rbins`)-1 length array containing the correlation function :math:`\\xi(r)` 
        computed in each of the bins defined by input `rbins`.
        
        :math:`1 + \\xi(r) \\equiv \\mathrm{DD} / \\mathrm{RR}`, if the 'Natural' 
        `estimator` is used, where  :math:`\\mathrm{DD}` is calculated by the pair 
        counter, and :math:`\\mathrm{RR}` is counted internally using 'analytic randoms' 
        if no `randoms` are passed as an argument (see notes for an explanation).
        
        If `sample2` is passed as input, three arrays of length len(`rbins`)-1 are 
        returned: :math:`\\xi_{11}(r)`, :math:`\\xi_{12}(r)`, :math:`\\xi_{22}(r)`,
        the autocorrelation of sample1, the cross-correlation between `sample1` and 
        `sample2`, and the autocorrelation of `sample2`.  If `do_auto` or `do_cross` is 
        set to False, the appropriate result(s) are returned.

    Notes
    -----
    Pairs are counted using the pair_counters.double_tree_pairs module.  This pair 
    counter is optimized to work on points distributed in a rectangular cuboid volume, 
    e.g. a simulation box.  This optimization restricts this function to work on 3-D 
    point distributions.
    
    If the points are distributed in a 'periodic box', then `randoms` are not necessary, 
    as the geometry is very simple, and the monte carlo integration that randoms are used 
    for in complex geometries can be done analytically.
    
    If the `period` argument is passed, points may not have any component of their 
    coordinates be negative.
    """
    
    #check input arguments using helper function
    sample1, rbins, sample2, randoms, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = _tpcf_process_args(sample1, rbins, sample2, randoms,\
                                                       period, do_auto, do_cross,\
                                                       estimator, num_threads,\
                                                       max_sample_size)
    
    
    def random_counts(sample1, sample2, randoms, rbins, period, PBCs, num_threads,\
                      do_RR, do_DR, _sample1_is_sample2):
        """
        Count random pairs.  There are three high level branches: 

            1. no PBCs w/ randoms.

            2. PBCs w/ randoms

            3. PBCs and analytical randoms

        There are also logical bits to do RR and DR pair counts, as not all estimators 
        need one or the other, and not doing these can save a lot of calculation.
        
        If no randoms are passes, calculate analytical randoms; otherwise, do it the old 
        fashioned way.
        """
        def nball_volume(R,k=3):
            """
            Calculate the volume of a n-shpere.  This is used for the analytical randoms.
            """
            return (np.pi**(k/2.0)/gamma(k/2.0+1.0))*R**k
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            RR = npairs(randoms, randoms, rbins, period=period, num_threads=num_threads)
            RR = np.diff(RR)
            D1R = npairs(sample1, randoms, rbins, period=period, num_threads=num_threads)
            D1R = np.diff(D1R)
            if _sample1_is_sample2: #calculating the cross-correlation
                D2R = None
            else:
                D2R = npairs(sample2, randoms, rbins, period=period, num_threads=num_threads)
                D2R = np.diff(D2R)
            
            return D1R, D2R, RR
        #PBCs and randoms.
        elif randoms is not None:
            if do_RR==True:
                RR = npairs(randoms, randoms, rbins, period=period, num_threads=num_threads)
                RR = np.diff(RR)
            else: RR=None
            if do_DR==True:
                D1R = npairs(sample1, randoms, rbins, period=period, num_threads=num_threads)
                D1R = np.diff(D1R)
            else: D1R=None
            if _sample1_is_sample2: #calculating the cross-correlation
                D2R = None
            else:
                if do_DR==True:
                    D2R = npairs(sample2, randoms, rbins, period=period,\
                                 num_threads=num_threads)
                    D2R = np.diff(D2R)
                else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif randoms is None:
            #do volume calculations
            dv = nball_volume(rbins) #volume of spheres
            dv = np.diff(dv) #volume of shells
            global_volume = period.prod() #sexy
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0]
            rho1 = N1/global_volume
            D1R = (N1)*(dv*rho1) #read note about pair counter
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if _sample1_is_sample2:
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_volume
                D2R = N2*(dv*rho2) #read note about pair counter
                #calculate the random-random pairs.
                #RR is only the RR for the cross-correlation when using analytical randoms
                #for the non-cross case, DR==RR (in analytical world).
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor)

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    
    def pair_counts(sample1, sample2, rbins, period, N_thread, do_auto, do_cross,\
                    _sample1_is_sample2):
        """
        Count data pairs.
        """
        if do_auto==True:
            D1D1 = npairs(sample1, sample1, rbins, period=period, num_threads=num_threads)
            D1D1 = np.diff(D1D1)
        else:
            D1D1=None
            D2D2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = npairs(sample1, sample2, rbins, period=period, num_threads=num_threads)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto==True:
                D2D2 = npairs(sample2, sample2, rbins, period=period, num_threads=num_threads)
                D2D2 = np.diff(D2D2)
            else: D2D2=None
        
        return D1D1, D1D2, D2D2
    
    
    #what needs to be done?
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
    
    #how many points are there? (for normalization purposes)
    if randoms is not None:
        N1 = len(sample1)
        NR = len(randoms)
        if _sample1_is_sample2:
            N2 = N1
        else:
            N2 = len(sample2)
    else: #this is taken care of in the analytical randoms case.
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    #count pairs
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, rbins, period,\
                                 num_threads, do_auto, do_cross, _sample1_is_sample2)
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, rbins, period,\
                                 PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2)
    
    #run through estimator and return relavent results
    if _sample1_is_sample2: #only do auto
        xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return xi_11
    else: #else, follow the user's input
        if (do_auto==True) & (do_cross==True): 
            xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            xi_22 = _TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return xi_11, xi_12, xi_22
        elif (do_cross==True):
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            return xi_12
        elif (do_auto==True):
            xi_11 = _TP_estimator(D1D1,D1R,D1R,N1,N1,NR,NR,estimator)
            xi_22 = _TP_estimator(D2D2,D2R,D2R,N2,N2,NR,NR,estimator)
            return xi_11


