"""
Module containing the `~halotools.mock_observables.tpcf` function used to 
calculate the two-point correlation function in 3d (aka galaxy clustering). 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import gamma
from warnings import warn

from .clustering_helpers import _tpcf_process_args
from .tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from .pair_counters import npairs_3d
##########################################################################################


__all__ = ['tpcf']
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR

def _random_counts(sample1, sample2, randoms, rbins, period, PBCs, num_threads,
    do_RR, do_DR, _sample1_is_sample2, approx_cell1_size,
    approx_cell2_size , approx_cellran_size):
    """
    Internal function used to random pairs during the calculation of the tpcf.  
    There are two high level branches:
        1. w/ or wo/ PBCs and randoms.
        2. PBCs and analytical randoms
    There is also control flow governing whether RR and DR pairs are counted, 
    as not all estimators need one or the other.
    
    Analytical counts are N**2*dv*rho, where dv is the volume of the spherical 
    shells, which is the correct volume to use for a continuous cubical volume with PBCs. 
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
    Internal function used calculate DD-pairs during the calculation of the tpcf.
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

    See also :ref:`galaxy_catalog_analysis_tutorial2` for example usage on a 
    mock galaxy catalog. 
    
    Parameters 
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are counted.
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points. 
        Passing ``sample2`` as an input permits the calculation of 
        the cross-correlation function. 
        Default is None, in which case only the 
        auto-correlation function will be calculated. 
    
    randoms : array_like, optional
        Nran x 3 array containing 3-D positions of randomly distributed points. 
        If no randoms are provided (the default option), 
        calculation of the tpcf can proceed using analytical randoms 
        (only valid for periodic boundary conditions).
    
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
        If set to None (the default option), PBCs are set to infinity, 
        in which case ``randoms`` must be provided. 
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    do_auto : boolean, optional
        Boolean determines whether the auto-correlation function will 
        be calculated and returned. Default is True. 
    
    do_cross : boolean, optional
        Boolean determines whether the cross-correlation function will 
        be calculated and returned. Only relevant when ``sample2`` is also provided. 
        Default is True for the case where ``sample2`` is provided, otherwise False. 
    
    estimator : string, optional
        Statistical estimator for the tpcf. 
        Options are 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
        Default is ``Natural``. 

    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed 
        using the python ``multiprocessing`` module. Default is 1 for a purely serial 
        calculation, in which case a multiprocessing Pool object will 
        never be instantiated. A string 'max' may be used to indicate that 
        the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        If sample size exeeds max_sample_size, 
        the sample will be randomly down-sampled such that the subsample 
        is equal to ``max_sample_size``. Default value is 1e6. 
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by how points 
        will be apportioned into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use Lbox/10 in each dimension, 
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
        Array storing the number of RR-counts calculated in advance during 
        a pre-processing phase. Must have the same length as *len(rbins)*. 
        If the ``RR_precomputed`` argument is provided, 
        you must also provide the ``NR_precomputed`` argument. 
        Default is None. 

    NR_precomputed : int, optional 
        Number of points in the random sample used to calculate ``RR_precomputed``.  
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
        
        If ``estimator`` is set to 'Natural'.  :math:`\\mathrm{DD}(r)` is the number
        of sample pairs with separations equal to :math:`r`, calculated by the pair 
        counter.  :math:`\\mathrm{RR}(r)` is the number of random pairs with separations 
        equal to :math:`r`, and is counted internally using "analytic randoms" if 
        ``randoms`` is set to None (see notes for an explanation), otherwise it is 
        calculated using the pair counter.
        
        If ``sample2`` is passed as input 
        (and if ``sample2`` is not exactly the same as ``sample1``), 
        then three arrays of length *len(rbins)-1* are returned:
        
        .. math::
            \\xi_{11}(r), \\xi_{12}(r), \\xi_{22}(r),
        
        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and 
        ``sample2``, and the autocorrelation of ``sample2``, respectively. 
        If ``do_auto`` or ``do_cross`` is set to False, 
        the appropriate sequence of results is returned.

    Notes
    -----
    Pairs are counted using `~halotools.mock_observables.npairs_3d`.  
        
    If the ``period`` argument is passed in, the ith coordinate of all points
    must be between 0 and period[i].
    
    For a higher-performance implementation of the tpcf function written in C, 
    see the Corrfunc code written by Manodeep Sinha, available at 
    https://github.com/manodeep/Corrfunc. 

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = Lbox
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array` 
    convenience function for this same purpose, which provides additional wrapper 
    behavior around `numpy.vstack` such as placing points into redshift-space. 
    
    >>> rbins = np.logspace(-2,-1,10)
    >>> xi = tpcf(coords, rbins, period=period)
    
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



