# -*- coding: utf-8 -*-

"""
Calculate one and two halo two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma
from .clustering_helpers import *
from .pair_counters.double_tree_pairs import npairs
from warnings import warn
##########################################################################################


__all__=['tpcf_one_two_halo_decomp']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def tpcf_one_two_halo_decomp(sample1, sample1_host_halo_id, rbins,
                             sample2=None, sample2_host_halo_id,
                             randoms=None, period=None,
                             do_auto=True, do_cross=True, estimator='Natural',
                             num_threads=1, max_sample_size=int(1e6),
                             approx_cell1_size = None, approx_cell2_size = None,
                             approx_cellran_size = None):
    """ 
    Calculate the real space one-halo and two-halo decomposed two-point correlation functions, :math:`\\xi^{1h}(r)`, :math:`\\xi^{2h}(r)`.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    sample1_host_halo_id: array_like
        *len(sample1)* integer array of host halo ids.
    
    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are 
        counted.
    
    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.
    
    sample2_host_halo_id: array_like, optional
        *len(sample2)* integer array of host halo ids.
    
    randoms : array_like, optional
        Npts x 3 array containing 3-D positions of points.  If no randoms are provided
        analytic randoms are used (only valid for periodic boundary conditions).
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox]*3.
    
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
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by which 
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
        will apportion the sample1 points into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use max(rbins) in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 

    approx_cell2_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for ``sample2``.  See comments for 
        ``approx_cell1_size`` for details. 
    
    approx_cellran_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for ``randoms``.  See comments for 
        ``approx_cell1_size`` for details. 
    
    Returns 
    -------
    correlation_functions : numpy.array
        Two *len(rbins)-1* length arrays containing the one and two halo correlation 
        functions, :math:`\\xi^{1h}(r)` and :math:`\\xi^{2h}(r)`, computed in each of the
        radial bins defined by input ``rbins``.
        
        .. math::
            `1 + \\xi(r) \\equiv \\mathrm{DD} / \\mathrm{RR}`,
        
        if the "Natural" ``estimator`` is used, where  :math:`\\mathrm{DD}` is calculated 
        by the pair counter, and :math:`\\mathrm{RR}` is counted internally using 
        "analytic randoms" if no ``randoms`` are passed as an argument 
        (see notes for an explanation).  If a different ``estimator`` is specified, the 
        appropiate formula is used.
        
        If ``sample2`` is passed as input, six arrays of length len(``rbins``)-1 are 
        returned:
        
        .. math::
            `\\xi^{1h}_{11}(r), \\xi^{2h}_{11}(r)`,
        .. math::
            `\\xi^{1h}_{12}(r), \\xi^{2h}_{12}(r)`,
        .. math::
            `\\xi^{1h}_{22}(r), \\xi^{2h}_{22}(r)`,
        
        the autocorrelation of one and two halo autocorrelation of sample1, 
        the one and two halo cross-correlation between ``sample1`` and ``sample2``,
        and the one and two halo autocorrelation of ``sample2``.  
        If ``do_auto`` or ``do_cross`` is set to False, only the appropriate result(s) 
        is returned.

    Notes
    -----
    Data-data pairs (:math:`DD`) are counted using the 
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.marked_npairs`,
    and random pairs (:math:`DR` and :math:`RR`) are counted using the 
    `~halotools.mock_observables.pair_counters.double_tree_pairs.npiars`.
    This pair counter is optimized to work on points distributed in a rectangular cuboid 
    volume, e.g. a simulation box.  This optimization restricts this function to work on 
    3-D point distributions.
    
    If the points are distributed in a continuous "periodic box", then ``randoms`` are not 
    necessary, as the geometry is very simple, and the monte carlo integration that 
    randoms are used for in complex geometries can be done analytically.
    
    If the ``period`` argument is passed, all points' :math:`i^{\rm th}` coordinate must 
    be between [0,``period[:math:`i`]``].
    """
    
    #check input arguments using clustering helper functions
    function_args = [sample1, sample1_host_halo_id, rbins, sample2, sample2_host_halo_id,
                     randoms, period, do_auto, do_cross, estimator, num_threads,
                     max_sample_size, approx_cell1_size, approx_cell2_size,
                     approx_cellran_size]
    
    #pass arguments in, and get out processed arguments, plus some control flow variables
    sample1, sample1_host_halo_id, rbins, sample2, sample2_host_halo_id, randoms, period,
    do_auto, do_cross, num_threads, _sample1_is_sample2, PBCs =
    _tpcf_one_two_halo_decomp_process_args(*function_args)
    
    
    def random_counts(sample1, sample2, randoms, rbins, period, PBCs, num_threads,\
                      do_RR, do_DR, _sample1_is_sample2, approx_cell1_size,\
                      approx_cell2_size , approx_cellran_size):
        """
        Count random pairs.  There are two high level branches:
            1. w/ or wo/ PBCs and randoms.
            2. PBCs and analytical randoms
        There are also logical bits to do RR and DR pair counts, as not all estimators
        need one or the other, and not doing these can save a lot of calculation.
        
        Analytical counts are N**2*dv*rho, where dv can is the volume of the spherical 
        shells, which is the correct volume to use for a continious cubic volume with PBCs
        """
        def nball_volume(R,k=3):
            """
            Calculate the volume of a n-shpere.
            This is used for the analytical randoms.
            """
            return (np.pi**(k/2.0)/gamma(k/2.0+1.0))*R**k
        
        #randoms provided, so calculate random pair counts.
        if randoms is not None:
            if do_RR==True:
                RR = npairs(randoms, randoms, rbins, period=period,
                            num_threads=num_threads,
                            approx_cell1_size=approx_cellran_size,
                            approx_cell2_size=approx_cellran_size)
                RR = np.diff(RR)
            else: RR=None
            if do_DR==True:
                D1R = npairs(sample1, randoms, rbins, period=period,
                             num_threads=num_threads,
                             approx_cell1_size=approx_cell1_size,
                             approx_cell2_size=approx_cellran_size
                             )
                D1R = np.diff(D1R)
            else: D1R=None
            if _sample1_is_sample2:
                D2R = None
            else:
                if do_DR==True:
                    D2R = npairs(sample2, randoms, rbins, period=period,
                                 num_threads=num_threads,
                                 approx_cell1_size=approx_cell2_size,
                                 approx_cell2_size=approx_cellran_size)
                    D2R = np.diff(D2R)
                else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif randoms is None:
            #do volume calculations
            dv = nball_volume(rbins) #volume of spheres
            dv = np.diff(dv) #volume of shells
            global_volume = period.prod() #volume of simulation
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0] #number of points in sample1
            rho1 = N1/global_volume #number density of points
            D1R = (N1)*(dv*rho1) #random counts are N**2*dv*rho
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if _sample1_is_sample2:
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_volume #number density of points
                D2R = N2*(dv*rho2)
                #calculate the random-random pairs.
                #RR is only the RR for the cross-correlation when using analytical randoms
                #for the non-cross case, DR==RR (in analytical world).
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor)

            return D1R, D2R, RR
    
    
    def pair_counts(sample1, sample2, rbins, period, num_threads,\
                    do_auto, do_cross, marks1, marks2, wfunc, _sample1_is_sample2)
        """
        Count weighted data pairs.
        """
        
        #add ones to weights, so returned value is return 1.0*1.0
        marks1 = np.vstack((marks1,np.ones(len(marks1)))).T
        marks2 = np.vstack((marks2,np.ones(len(marks2)))).T
        
        if do_auto==True:
            D1D1 = marked_npairs(sample1, sample1, rbins,\
                                 weights1=marks1, weights2=marks1,\
                                 wfunc = wfunc,\
                                 period=period, num_threads=num_threads)
            D1D1 = np.diff(D1D1)
        else:
            D1D1=None
            D2D2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = marked_npairs(sample1, sample2, rbins,\
                                     weights1=marks1, weights2=marks2,\
                                     wfunc = wfunc,\
                                     period=period, num_threads=num_threads)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto==True:
                D2D2 = marked_npairs(sample2, sample2, rbins,\
                                     weights1=marks2, weights2=marks2,\
                                     wfunc = wfunc,\
                                     period=period, num_threads=num_threads)
                D2D2 = np.diff(D2D2)
            else: D2D2=None

        return D1D1, D1D2, D2D2
    
    #What needs to be done?
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
    
    #calculate 1-halo pairs
    wfunc=3
    one_halo_D1D1,one_halo_D1D2, one_halo_D2D2 =\
        marked_pair_counts(sample1, sample2, rbins, period, num_threads,\
                           do_auto, do_cross, marks1, marks2, wfunc, _sample1_is_sample2)
    
    #calculate 2-halo pairs 
    wfunc=13
    two_halo_D1D1,two_halo_D1D2, two_halo_D2D2 =\
        marked_pair_counts(sample1, sample2, rbins, period, num_threads,\
                           do_auto, do_cross, marks1, marks2, wfunc, _sample1_is_sample2)
    
    #count random pairs
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, rbins, period,
                                 PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2,
                                 approx_cell1_size,approx_cell2_size,approx_cellran_size)
    
    #check to see if any of the random counts contain 0 pairs.
    if D1R is not None:
        if np.any(D1R==0):
            msg = ("sample1 cross randoms has radial bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    if D2R is not None:
        if np.any(D2R==0):
            msg = ("sample2 cross randoms has radial bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    if RR is not None:
        if np.any(RR==0):
            msg = ("randoms cross randoms has radial bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    
    #run results through the estimator and return relavent/user specified results.
    if _sample1_is_sample2:
        one_halo_xi_11 = _TP_estimator(one_halo_D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        two_halo_xi_11 = _TP_estimator(two_halo_D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return one_halo_xi_11, two_halo__xi_11
    else:
        if (do_auto==True) & (do_cross==True): 
            one_halo_xi_11 = _TP_estimator(one_halo_D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            one_halo_xi_12 = _TP_estimator(one_halo_D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            one_halo_xi_22 = _TP_estimator(one_halo_D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            two_halo_xi_11 = _TP_estimator(two_halo_D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            two_halo_xi_12 = _TP_estimator(two_halo_D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            two_halo_xi_22 = _TP_estimator(two_halo_D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return one_halo_xi_11, two_halo_xi_11, one_halo_xi_12,\
                   two_halo_xi_12, one_halo_xi_22, two_halo_xi_22
        elif (do_cross==True):
            one_halo_xi_12 = _TP_estimator(one_halo_D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            two_halo_xi_12 = _TP_estimator(two_halo_D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            return one_halo_xi_12, two_halo_xi_12
        elif (do_auto==True):
            one_halo_xi_11 = _TP_estimator(one_halo_D1D1,D1R,D1R,N1,N1,NR,NR,estimator)
            one_halo_xi_22 = _TP_estimator(one_halo_D2D2,D2R,D2R,N2,N2,NR,NR,estimator)
            two_halo_xi_11 = _TP_estimator(two_halo_D1D1,D1R,D1R,N1,N1,NR,NR,estimator)
            two_halo_xi_22 = _TP_estimator(two_halo_D2D2,D2R,D2R,N2,N2,NR,NR,estimator)
            return one_halo_xi_11, two_halo_xi_11, one_halo_xi_22, two_halo_xi_22


