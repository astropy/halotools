# -*- coding: utf-8 -*-

"""
Calculate angular two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma
from .clustering_helpers import *
from ..utils.spherical_geometry import *
from warnings import warn
from .pair_counters.double_tree_pairs import npairs
##########################################################################################


__all__=['angular_tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def angular_tpcf(sample1, theta_bins, sample2=None, randoms=None,
                 do_auto=True, do_cross=True, estimator='Natural', num_threads=1,
                 max_sample_size=int(1e6)):
    """ 
    Calculate the angular two-point correlation function, :math:`w(\\theta)`.
    
    Example calls to this function appear in the documentation below. For thorough 
    documentation of all features, see :ref:`angular_tpcf_usage_tutorial`. 
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 2 numpy array containing ra,dec positions of points in degrees.
    
    theta_bins : array_like
        array of boundaries defining the angular distance bins in which pairs are 
        counted.
    
    sample2 : array_like, optional
        Npts x 2 array containing ra,dec positions of points in degrees.
    
    randoms : array_like, optional
        Npts x 2 array containing ra,dec positions of points in degrees.  If no randoms 
        are provided analytic randoms are used (only valid for for continious all-sky 
        converaeg).
    
    do_auto : boolean, optional
        do auto-correlation(s)?
    
    do_cross : boolean, optional
        do cross-correlation?
    
    estimator : string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. If 
        sample size exeeds max_sample_size, the sample will be randomly down-sampled such
        that the subsample is equal to ``max_sample_size``. 
    
    Returns 
    -------
    correlation_function(s) : numpy.array
        *len(theta_bins)-1* length array containing the correlation function 
        :math:`w(\\theta)` computed in each of the bins defined by input ``theta_bins``.
        
        .. math::
            1 + w(\\theta) \\equiv \\mathrm{DD}(\\theta) / \\mathrm{RR}(\\theta),
        
        if ``estimator`` is set to 'Natural'.  :math:`\\mathrm{DD}(\\theta)` is the number
        of sample pairs with seperations equal to :math:`\\theta`, calculated by the pair 
        counter.  :math:`\\mathrm{RR}(\\theta)` is the number of random pairs with 
        seperations equal to :math:`\\theta`, and is counted internally using 
        "analytic randoms" if ``randoms`` is set to None (see notes for an explanation), 
        otherwise it is calculated using the pair counter.
        
        If ``sample2`` is passed as input (and not exactly the same as ``sample1``), 
        three arrays of length *len(theta_bins)-1* are returned:
        
        .. math::
            w_{11}(\\theta), w_{12}(\\theta), w_{22}(\\theta),
        
        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and 
        ``sample2``, and the autocorrelation of ``sample2``, respectively. If 
        ``do_auto`` or ``do_cross`` is set to False, the appropriate result(s) are 
        returned.

    Notes
    -----
    Pairs are counted using `~halotools.mock_observables.pair_counters.npairs`.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
   
    >>> angular_coords = sample_spherical_surface(Npts)
    
    >>> theta_bins = np.logspace(-2,1,10)
    >>> w = angular_tpcf(angular_coords, theta_bins)
    
    The result should be consistent with zero correlation at all *theta* within 
    statistical errors
    """
    
    #check input arguments using clustering helper functions
    function_args = [sample1, theta_bins, sample2, randoms, do_auto, do_cross,\
                     estimator, num_threads, max_sample_size]
    
    #pass arguments in, and get out processed arguments, plus some control flow variables
    sample1, theta_bins, sample2, randoms, do_auto, do_cross, num_threads,\
    _sample1_is_sample2 = _angular_tpcf_process_args(*function_args)
    
    #convert angular bins to coord lengths on a unit sphere
    chord_bins  = chord_to_cartesian(theta_bins, radians=False)
    
    #convert samples and randoms to cartesian coordinates (x,y,z) on a unit sphere
    x,y,z = spherical_to_cartesian(sample1[:,0], sample1[:,1])
    sample1 = np.vstack((x,y,z)).T
    if _sample1_is_sample2:
        sample2 = sample1
    else:
        x,y,z = spherical_to_cartesian(sample2[:,0], sample2[:,1])
        sample2 = np.vstack((x,y,z)).T
    if randoms is not None:
        x,y,z = spherical_to_cartesian(randoms[:,0], randoms[:,1])
        randoms = np.vstack((x,y,z)).T
    
    def random_counts(sample1, sample2, randoms, chord_bins, num_threads,\
                      do_RR, do_DR, _sample1_is_sample2):
        """
        Count random pairs.
        """
        def area_spherical_cap(chord):
            """
            Calculate the area of a spherical cap on a unit sphere given the chord length
            """
            h = 1.0 - np.sqrt(1.0-chord**2)
            return np.pi*(chord**2+h**2)
        
        #randoms provided, so calculate random pair counts.
        if randoms is not None:
            if do_RR==True:
                RR = npairs(randoms, randoms, chord_bins,
                            num_threads=num_threads)
                RR = np.diff(RR)
            else: RR=None
            if do_DR==True:
                D1R = npairs(sample1, randoms, chord_bins,
                             num_threads=num_threads)
                D1R = np.diff(D1R)
            else: D1R=None
            if _sample1_is_sample2:
                D2R = None
            else:
                if do_DR==True:
                    D2R = npairs(sample2, randoms, chord_bins,
                                 num_threads=num_threads)
                    D2R = np.diff(D2R)
                else: D2R=None
            return D1R, D2R, RR
        elif randoms is None:
            #do area calculations
            da = area_spherical_cap(chord_bins)
            da = np.diff(da)
            global_area = 4.0*np.pi #surface area of a unit sphere
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0] #number of points in sample1
            rho1 = N1/global_area #number density of points
            D1R = (N1)*(da*rho1) #random counts are N**2*dv*rho
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if _sample1_is_sample2:
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_area #number density of points
                D2R = N2*(da*rho2)
                #calculate the random-random pairs.
                #RR is only the RR for the cross-correlation when using analytical randoms
                #for the non-cross case, DR==RR (in analytical world).
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor)

            return D1R, D2R, RR
    
    def pair_counts(sample1, sample2, chord_bins, N_thread, do_auto, do_cross,\
                    _sample1_is_sample2):
        """
        Count data-data pairs.
        """
        
        if do_auto==True:
            D1D1 = npairs(sample1, sample1, chord_bins, num_threads=num_threads)
            D1D1 = np.diff(D1D1)
        else:
            D1D1=None
            D2D2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = npairs(sample1, sample2, chord_bins, num_threads=num_threads)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto==True:
                D2D2 = npairs(sample2, sample2, rbins, num_threads=num_threads)
                D2D2 = np.diff(D2D2)
            else: D2D2=None
        
        return D1D1, D1D2, D2D2
    
    # What needs to be done?
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
    
    # How many points are there (for normalization purposes)?
    if randoms is not None:
        N1 = len(sample1)
        NR = len(randoms)
        if _sample1_is_sample2:
            N2 = N1
        else:
            N2 = len(sample2)
    else: # This is taken care of in the analytical case.  See comments in random_pairs().
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    #count data pairs
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, chord_bins,
                                 num_threads, do_auto, do_cross, _sample1_is_sample2)
    #count random pairs
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, theta_bins,
                                 num_threads, do_RR, do_DR, _sample1_is_sample2)
    
    #check to see if any of the random counts contain 0 pairs.
    if D1R is not None:
        if np.any(D1R==0):
            msg = ("sample1 cross randoms has theta bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    if D2R is not None:
        if np.any(D2R==0):
            msg = ("sample2 cross randoms has theta bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    if RR is not None:
        if np.any(RR==0):
            msg = ("randoms cross randoms has theta bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    
    #run results through the estimator and return relavent/user specified results.
    if _sample1_is_sample2:
        xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return xi_11
    else:
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
            return xi_11, xi_22

