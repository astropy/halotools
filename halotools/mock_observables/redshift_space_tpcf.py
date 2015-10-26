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
from .pair_counters.rect_cuboid_pairs import xy_z_npairs
##########################################################################################


__all__=['redshift_space_tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def redshift_space_tpcf(sample1, rp_bins, pi_bins, sample2=None, randoms=None,\
                        period=None, do_auto=True, do_cross=True, estimator='Natural',\
                        num_threads=1, max_sample_size=int(1e6)):
    """ 
    Calculate the redshift space correlation function, :math:`\\xi(r_{p}, \\pi)`.
    
    The first two dimensions define the plane for perpendicular distances.  The third 
    dimension is used for parallel distances.  i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate.  This is the distant observer approximation.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points. 
    
    rp_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    pi_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    sample2 : array_like, optional
        Npts x 3 numpy array containing 3-D positions of points.
    
    randoms : array_like, optional
        Nran x 3 numpy array containing 3-D positions of points.  If no randoms are 
        provided analytic randoms are used (only valid for periodic boundary conditions).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    estimator : string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    do_auto : boolean, optional
        do auto-correlation?  Default is True.
    
    do_cross : boolean, optional
        do cross-correlation?  Default is True.
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exceeds `max_sample_size`, the sample will be randomly down-sampled 
        such that the subsample is equal to `max_sample_size`. 

    Returns 
    -------
    correlation_function : array_like
        ndarray containing correlation function :math:`\\xi(r_{p}, \\pi)` computed in each 
        of the len(`rp_bins`)-1 by len(`pi_bins`)-1 bins defined by input `rp_bins` and 
        `pi_bins`.

        :math:`1 + \\xi(r_{p},\\pi) = \\mathrm{DD} / \\mathrm{RR}`, is the 'Natural' 
        `estimator` is used, where :math:`\\mathrm{DD}` is calculated by the pair 
        counter, and :math:`\\mathrm{RR}` is counted internally analytic randoms if no 
        `randoms` are passed as an argument.

        If sample2 is passed as input, three ndarrays of shape 
        (len(`rp_bins`)-1,len(`pi_bins`)-1) are returned: 
        :math:`\\xi_{11}(rp, \\pi)`, :math:`\\xi_{12}(r_{p},\\pi)`, 
        :math:`\\xi_{22}(r_{p},\\pi)`,
        
        The autocorrelation of` sample1`, the cross-correlation between `sample1` 
        and `sample2`, and the autocorrelation of `sample2`.  If `do_auto` or `do_cross` 
        is set to False, the appropriate result(s) is not returned.

    """
    
    sample1, rp_bins, pi_bins, sample2, randoms, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = _redshift_space_tpcf_process_args(sample1, rp_bins,\
        pi_bins, sample2, randoms, period, do_auto, do_cross, estimator, num_threads,\
        max_sample_size)
    
    
    def random_counts(sample1, sample2, randoms, rp_bins, pi_bins, period,\
                      PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2):
        """
        Count random pairs.  There are three high level branches: 

            1. no PBCs w/ randoms.

            2. PBCs w/ randoms
            
            3. PBCs and analytical randoms

        There are also logical bits to do RR and DR pair counts, as not all estimators 
        need one or the other, and not doing these can save a lot of calculation.
        """
        def cylinder_volume(R,h):
            """
            Calculate the volume of a cylinder(s), used for the analytical randoms.
            """
            return pi*np.outer(R**2.0,h)
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            RR = xy_z_npairs(randoms, randoms, rp_bins, pi_bins, period=period,\
                             num_threads=num_threads)
            RR = np.diff(np.diff(RR,axis=0),axis=1)
            D1R = xy_z_npairs(sample1, randoms, rp_bins, pi_bins, period=period,\
                              num_threads=num_threads)
            D1R = np.diff(np.diff(D1R,axis=0),axis=1)
            if _sample1_is_sample2: #calculating the cross-correlation
                D2R = None
            else:
                D2R = xy_z_npairs(sample2, randoms, rp_bins, pi_bins, period=period,\
                                  num_threads=num_threads)
                D2R = np.diff(np.diff(D2R,axis=0),axis=1)
            
            return D1R, D2R, RR
        #PBCs and randoms.
        elif randoms is not None:
            if do_RR==True:
                RR = xy_z_npairs(randoms, randoms, rp_bins, pi_bins, period=period,\
                                 num_threads=num_threads)
                RR = np.diff(np.diff(RR,axis=0),axis=1)
            else: RR=None
            if do_DR==True:
                D1R = xy_z_npairs(sample1, randoms, rp_bins, pi_bins, period=period,\
                                  num_threads=num_threads)
                D1R = np.diff(np.diff(D1R,axis=0),axis=1)
            else: D1R=None
            if _sample1_is_sample2: #calculating the cross-correlation
                D2R = None
            else:
                if do_DR==True:
                    D2R = xy_z_npairs(sample2, randoms, rp_bins, pi_bins, period=period,\
                                      num_threads=num_threads)
                    D2R = np.diff(np.diff(D2R,axis=0),axis=1)
                else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif randoms is None:
            #do volume calculations
            dv = cylinder_volume(rp_bins,2.0*pi_bins) #volume of spheres
            dv = np.diff(np.diff(dv, axis=0),axis=1) #volume of annuli
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
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor) #RR is only the RR for the cross-correlation.

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    def pair_counts(sample1, sample2, rp_bins, pi_bins, period,\
                    N_thread, do_auto, do_cross, _sample1_is_sample2):
        """
        Count data pairs.
        """
        D1D1 = xy_z_npairs(sample1, sample1, rp_bins, pi_bins, period=period,\
                           num_threads=num_threads)
        D1D1 = np.diff(np.diff(D1D1,axis=0),axis=1)
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            D1D2 = xy_z_npairs(sample1, sample2, rp_bins, pi_bins, period=period,\
                               num_threads=num_threads)
            D1D2 = np.diff(np.diff(D1D2,axis=0),axis=1)
            D2D2 = xy_z_npairs(sample2, sample2, rp_bins, pi_bins, period=period,\
                               num_threads=num_threads)
            D2D2 = np.diff(np.diff(D2D2,axis=0),axis=1)

        return D1D1, D1D2, D2D2
    
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
              
    #how many points are there? (for normalization purposes)
    if randoms is not None:
        N1 = len(sample1)
        NR = len(randoms)
        if _sample1_is_sample2:
            N2 = N1
        else:
            N2 = len(sample2)
    else: 
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    #count pairs
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, rp_bins, pi_bins, period,\
                                 num_threads, do_auto, do_cross, _sample1_is_sample2)
    
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, rp_bins, pi_bins, period,\
                                 PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2)
    
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
            return xi_11


