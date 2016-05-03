# -*- coding: utf-8 -*-

"""
Calculate two point correlation functions.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
####import modules########################################################################
import numpy as np
from math import gamma
from .clustering_helpers import _tpcf_process_args
from .tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from .pair_counters import npairs_3d
from warnings import warn
##########################################################################################


__all__=['tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR

def _random_counts(sample1, sample2, randoms, rbins, period, PBCs, num_threads,
    do_RR, do_DR, _sample1_is_sample2, approx_cell1_size,
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
        if do_RR is True:
            RR = npairs_3d(randoms, randoms, rbins, period=period,
                        num_threads=num_threads,
                        approx_cell1_size=approx_cellran_size,
                        approx_cell2_size=approx_cellran_size)
            RR = np.diff(RR)
        else: RR=None
        if do_DR is True:
            D1R = npairs_3d(sample1, randoms, rbins, period=period,
                         num_threads=num_threads,
                         approx_cell1_size=approx_cell1_size,
                         approx_cell2_size=approx_cellran_size
                         )
            D1R = np.diff(D1R)
        else: D1R=None
        if _sample1_is_sample2:
            D2R = None
        else:
            if do_DR is True:
                D2R = npairs_3d(sample2, randoms, rbins, period=period,
                             num_threads=num_threads,
                             approx_cell1_size=approx_cell2_size,
                             approx_cell2_size=approx_cellran_size)
                D2R = np.diff(D2R)
            else: D2R=None
        
        return D1R, D2R, RR
    
    #PBCs and no randoms--calculate randoms analytically.
    elif randoms is None:
        
        #set the number of randoms equal to the number of points in sample1
        NR = len(sample1)
        
        #do volume calculations
        v = nball_volume(rbins) #volume of spheres
        dv = np.diff(v) #volume of shells
        global_volume = period.prod() #volume of simulation
        
        #calculate randoms for sample1
        N1 = np.shape(sample1)[0] #number of points in sample1
        rho1 = N1/global_volume #number density of points
        D1R = (NR)*(dv*rho1) #random counts are N**2*dv*rho
        
        #calculate randoms for sample2
        N2 = np.shape(sample2)[0] #number of points in sample2
        rho2 = N2/global_volume #number density of points
        D2R = (NR)*(dv*rho2) #random counts are N**2*dv*rho
        
        #calculate the random-random pairs.
        rhor = (NR**2)/global_volume
        RR = (dv*rhor)
        
        return D1R, D2R, RR

def _pair_counts(sample1, sample2, rbins, 
    period, num_threads, do_auto, do_cross,
    _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
    """
    Count data-data pairs.
    """
    if do_auto is True:
        D1D1 = npairs_3d(sample1, sample1, rbins, period=period, 
            num_threads=num_threads,
            approx_cell1_size=approx_cell1_size,
            approx_cell2_size=approx_cell1_size)
        D1D1 = np.diff(D1D1)
    else:
        D1D1=None
        D2D2=None
    
    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_3d(sample1, sample2, rbins, period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size)
            D1D2 = np.diff(D1D2)
        else: D1D2=None
        if do_auto is True:
            D2D2 = npairs_3d(sample2, sample2, rbins, period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell2_size,
                approx_cell2_size=approx_cell2_size)
            D2D2 = np.diff(D2D2)
        else: D2D2=None
    
    return D1D1, D1D2, D2D2


def tpcf(sample1, rbins, sample2=None, randoms=None, period=None,
    do_auto=True, do_cross=True, estimator='Natural', num_threads=1,
    max_sample_size=int(1e6), approx_cell1_size = None,
    approx_cell2_size = None, approx_cellran_size = None, 
    RR_precomputed = None, NR_precomputed = None):
    """ 
    Calculate the real space two-point correlation function, :math:`\\xi(r)`.
    
    Example calls to this function appear in the documentation below. 
    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` and ``sample2`` arguments.   

    See also :ref:`galaxy_catalog_analysis_tutorial2`. 
    
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
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by how points 
        will be apportioned into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use *max(rbins)* in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 

    approx_cell2_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details. 
    
    approx_cellran_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for randoms.  See comments for 
        ``approx_cell1_size`` for details. 

    RR_precomputed : array_like, optional 
        Array storing the number of previously calculated RR-counts. 
        Must have the same length as *len(rbins)*. 
        If the ``RR_precomputed`` argument is provided, 
        you must also provide the ``NR_precomputed`` argument. 
        Default is None. 

    NR_precomputed : int, optional 
        Number of points in the random sample used to calculate 
        ``RR_precomputed``.  
        If the ``NR_precomputed`` argument is provided, 
        you must also provide the ``RR_precomputed`` argument. 
        Default is None. 

    Returns 
    -------
    correlation_function(s) : numpy.array
        *len(rbins)-1* length array containing the correlation function :math:`\\xi(r)` 
        computed in each of the bins defined by input ``rbins``.
        
        .. math::
            1 + \\xi(r) \\equiv \\mathrm{DD}(r) / \\mathrm{RR}(r),
        
        if ``estimator`` is set to 'Natural'.  :math:`\\mathrm{DD}(r)` is the number
        of sample pairs with seperations equal to :math:`r`, calculated by the pair 
        counter.  :math:`\\mathrm{RR}(r)` is the number of random pairs with seperations 
        equal to :math:`r`, and is counted internally using "analytic randoms" if 
        ``randoms`` is set to None (see notes for an explanation), otherwise it is 
        calculated using the pair counter.
        
        If ``sample2`` is passed as input (and not exactly the same as ``sample1``), 
        three arrays of length *len(rbins)-1* are returned:
        
        .. math::
            \\xi_{11}(r), \\xi_{12}(r), \\xi_{22}(r),
        
        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and 
        ``sample2``, and the autocorrelation of ``sample2``, respectively. If 
        ``do_auto`` or ``do_cross`` is set to False, the appropriate result(s) are 
        returned.

    Notes
    -----
    Pairs are counted using 
    `~halotools.mock_observables.npairs_3d`.  This pair counter is optimized 
    to work on points distributed in a rectangular cuboid volume, e.g. a simulation box.  
    This optimization restricts this function to work on 3-D point distributions.
    
    If the points are distributed in a continuous "periodic box", then ``randoms`` are not 
    necessary, as the geometry is very simple, and the monte carlo integration that 
    randoms are used for in complex geometries can be done analytically.
    
    If the ``period`` argument is passed in, all points' ith coordinate 
    must be between 0 and period[i].
    
    For a higher-performance implementation of the tpcf function, 
    see the Corrfunc code written by Manodeep Sinha, available at 
    https://github.com/manodeep/Corrfunc. 

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    >>> rbins = np.logspace(-2,-1,10)
    >>> xi = tpcf(coords, rbins, period=period)
    
    The result should be consistent with zero correlation at all *r* within 
    statistical errors

    See also 
    --------
    :ref:`galaxy_catalog_analysis_tutorial2`
    """
    
    #check input arguments using clustering helper functions
    function_args = (sample1, rbins, sample2, randoms, period, 
        do_auto, do_cross, estimator, num_threads, max_sample_size, 
        approx_cell1_size, approx_cell2_size, approx_cellran_size, 
        RR_precomputed, NR_precomputed)
    
    #pass arguments in, and get out processed arguments, plus some control flow variables
    (sample1, rbins, sample2, randoms, period, 
        do_auto, do_cross, num_threads,
        _sample1_is_sample2, PBCs, 
        RR_precomputed, NR_precomputed) = _tpcf_process_args(*function_args)
    
    #Below we define functions to count data-data pairs and random pairs.
    #After that, we get to work. The pair counting functions here actually call outside
    #pair counters that are highly optimized. Beware that the control flow inside 
    #these functions here can look a bit complicated, but don't des-pair!
    
    
    # What needs to be done?
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
    if RR_precomputed is not None: do_RR = False

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    if randoms is not None:
        NR = len(randoms)
    else:
        #set the number of randoms equal to the number of points in sample1
        #this is arbitrarily set, but must remain consistent!
        if NR_precomputed is not None:
            NR = NR_precomputed
        else:
            NR = N1

    #count data pairs
    D1D1,D1D2,D2D2 = _pair_counts(sample1, sample2, rbins, period,
        num_threads, do_auto, do_cross, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size)
    #count random pairs
    D1R, D2R, RR = _random_counts(sample1, sample2, randoms, rbins, 
        period, PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size, approx_cellran_size)
    if RR_precomputed is not None: RR = RR_precomputed
    
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
        xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return xi_11
    else:
        if (do_auto is True) & (do_cross is True):
            xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            xi_22 = _TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return xi_11, xi_12, xi_22
        elif (do_cross is True):
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            return xi_12
        elif (do_auto is True):
            xi_11 = _TP_estimator(D1D1,D1R,D1R,N1,N1,NR,NR,estimator)
            xi_22 = _TP_estimator(D2D2,D2R,D2R,N2,N2,NR,NR,estimator)
            return xi_11, xi_22



