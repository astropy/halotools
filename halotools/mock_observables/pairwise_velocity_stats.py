# -*- coding: utf-8 -*-

"""
Calculate pairwise velocity statistics.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .pair_counters import velocity_marked_npairs_xy_z
from .pairwise_velocity_helpers import (_pairwise_velocity_stats_process_args, 
    _process_radial_bins, _process_rp_bins)

from .pair_counters import velocity_marked_npairs_3d

__all__ = ('mean_radial_velocity_vs_r', 'radial_pvd_vs_r',
    'mean_los_velocity_vs_rp', 'los_pvd_vs_rp')
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero


def mean_radial_velocity_vs_r(sample1, velocities1, rbins,
    sample2=None, velocities2=None,
    period=None, do_auto=True, do_cross=True,
    num_threads=1, max_sample_size=int(1e6),
    approx_cell1_size = None, approx_cell2_size = None):
    """ 
    Calculate the mean pairwise velocity, :math:`\\bar{v}_{12}(r)`.
    
    Example calls to this function appear in the documentation below. 
    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` and ``sample2`` arguments.   

    See also :ref:`galaxy_catalog_analysis_tutorial6`. 
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing the 3-D positions of points.
    
    velocities1 : array_like
        N1pts x 3 array containing the 3-D components of the velocities.
    
    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are 
        counted.
    
    sample2 : array_like, optional
        N2pts x 3 array containing the 3-D positions of points.
        
    velocities2 : array_like, optional
        N2pts x 3 array containing the 3-D components of the velocities.
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].
    
    do_auto : boolean, optional
        caclulate the auto-pairwise velocities?
    
    do_cross : boolean, optional
        caclulate the cross-pairwise velocities?
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled
        such that the subsample is equal to max_sample_size.
    
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
        Analogous to ``approx_cell1_size``, but for `sample2`.  See comments for 
        ``approx_cell1_size`` for details. 
    
    Returns 
    -------
    v_12 : numpy.array
        *len(rbins)-1* length array containing the mean pairwise velocity, 
        :math:`\\bar{v}_{12}(r)`, computed in each of the bins defined by ``rbins``.
    
    Notes
    -----
    The pairwise velocity, :math:`v_{12}(r)`, is defined as:
    
    .. math::
        v_{12}(r) = \\vec{v}_{\\rm 1, pec} \\cdot \\vec{r}_{12}-\\vec{v}_{\\rm 2, pec} \\cdot \\vec{r}_{12}
    
    where :math:`\\vec{v}_{\\rm 1, pec}` is the peculiar velocity of object 1, and
    :math:`\\vec{r}_{12}` is the radial vector connecting object 1 and 2.
    
    :math:`\\bar{v}_{12}(r)` is the mean of that quantity calculated in radial bins.
    
    Pairs and radial velocities are calculated using 
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs`.

    For radial separation bins in which there are zero pairs, function returns zero. 
    
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
    
    We will do the same to get a random set of peculiar velocities.
    
    >>> vx = np.random.random(Npts)-0.5
    >>> vy = np.random.random(Npts)-0.5
    >>> vz = np.random.random(Npts)-0.5
    >>> velocities = np.vstack((vx,vy,vz)).T
    
    >>> rbins = np.logspace(-2,-1,10)
    >>> v_12 = mean_radial_velocity_vs_r(coords, velocities, rbins, period=period)
    
    See also 
    --------
    :ref:`galaxy_catalog_analysis_tutorial6`

    """
    
    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    
    sample1, velocities1, sample2, velocities2, period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs = _pairwise_velocity_stats_process_args(*function_args)
    
    rbins = _process_radial_bins(rbins, period, PBCs)
    
    #create marks for the marked pair counter.
    marks1 = np.vstack((sample1.T, velocities1.T)).T
    marks2 = np.vstack((sample2.T, velocities2.T)).T
    
    def marked_pair_counts(sample1, sample2, rbins, period, num_threads,
        do_auto, do_cross, marks1, marks2,
        weight_func_id, _sample1_is_sample2, approx_cell1_size,approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """
        
        if do_auto is True:
            D1D1, dummy, N1N1 = velocity_marked_npairs_3d(
                sample1, sample1, rbins,
                weights1=marks1, weights2=marks1,
                weight_func_id = weight_func_id,
                period=period, num_threads=num_threads,
                approx_cell1_size = approx_cell1_size,
                approx_cell2_size = approx_cell1_size)
            D1D1 = np.diff(D1D1)
            N1N1 = np.diff(N1N1)
        else:
            D1D1=None
            D2D2=None
            N1N1=None
            N2N2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
        else:
            if do_cross is True:
                D1D2, dummy, N1N2 = velocity_marked_npairs_3d(
                    sample1, sample2, rbins,
                    weights1=marks1, weights2=marks2,
                    weight_func_id = weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell1_size,
                    approx_cell2_size = approx_cell2_size)
                D1D2 = np.diff(D1D2)
                N1N2 = np.diff(N1N2)
            else: 
                D1D2=None
                N1N2=None
            if do_auto is True:
                D2D2, dummy, N2N2 = velocity_marked_npairs_3d(
                    sample2, sample2, rbins,
                    weights1=marks2, weights2=marks2,
                    weight_func_id = weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell2_size,
                    approx_cell2_size = approx_cell2_size)
                D2D2 = np.diff(D2D2)
                N2N2 = np.diff(N2N2)
            else: 
                D2D2=None
                N2N2=None
    
        return D1D1, D1D2, D2D2, N1N1, N1N2, N2N2
    
    
    #count the sum of radial velocities and number of pairs
    weight_func_id = 11
    V1V1,V1V2,V2V2, N1N1,N1N2,N2N2 =\
        marked_pair_counts(sample1, sample2, rbins, period,
            num_threads, do_auto, do_cross,
            marks1, marks2, weight_func_id,
            _sample1_is_sample2,
            approx_cell1_size, approx_cell2_size)

    #return results: the sum of radial velocities divided by the number of pairs
    if _sample1_is_sample2:
        M_11 = V1V1/N1N1
        return np.where(np.isfinite(M_11), M_11, 0.)
    else:
        if (do_auto is True) & (do_cross is True): 
            M_11 = V1V1/N1N1
            M_12 = V1V2/N1N2
            M_22 = V2V2/N2N2
            return (np.where(np.isfinite(M_11), M_11, 0.), 
                np.where(np.isfinite(M_12), M_12, 0.), np.where(np.isfinite(M_22), M_22, 0.))
        elif do_cross is True:
            M_12 = V1V2/N1N2
            return np.where(np.isfinite(M_12), M_12, 0.)
        elif (do_auto is True):
            M_11 = V1V1/N1N1
            M_22 = V2V2/N2N2 
            return np.where(np.isfinite(M_11), M_11, 0.), np.where(np.isfinite(M_22), M_22, 0.)


def radial_pvd_vs_r(sample1, velocities1, rbins, sample2=None,
    velocities2=None, period=None, do_auto=True, do_cross=True,
    num_threads=1, max_sample_size=int(1e6),
    approx_cell1_size = None,approx_cell2_size = None):
    """
    Calculate the pairwise velocity dispersion (PVD), :math:`\\sigma_{12}(r)`.
    
    Example calls to this function appear in the documentation below.
    
    See also :ref:`galaxy_catalog_analysis_tutorial7`. 

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    velocities1 : array_like
        len(sample1) array of velocities.
    
    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are 
        counted.
    
    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.
        
    velocities2 : array_like, optional
        len(sample12) array of velocities.
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].
    
    do_auto : boolean, optional
        caclulate the auto-pairwise velocities?
    
    do_cross : boolean, optional
        caclulate the cross-pairwise velocities?
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled
        such that the subsample is equal to max_sample_size.
    
    Returns 
    -------
    sigma_12 : numpy.array
        *len(rbins)-1* length array containing the dispersion of the pairwise velocity, 
        :math:`\\sigma_{12}(r)`, computed in each of the bins defined by ``rbins``.
    
    Notes
    -----
    The pairwise velocity, :math:`v_{12}(r)`, is defined as:
    
    .. math::
        v_{12}(r) = \\vec{v}_{\\rm 1, pec} \\cdot \\vec{r}_{12}-\\vec{v}_{\\rm 2, pec} \\cdot \\vec{r}_{12}
    
    where :math:`\\vec{v}_{\\rm 1, pec}` is the peculiar velocity of object 1, and
    :math:`\\vec{r}_{12}` is the radial vector connecting object 1 and 2.
    
    :math:`\\sigma_{12}(r)` is the standard deviation of this quantity in radial bins.
    
    Pairs and radial velocities are calculated using 
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs`.
    
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
    
    We will do the same to get a random set of peculiar velocities.
    
    >>> vx = np.random.random(Npts)-0.5
    >>> vy = np.random.random(Npts)-0.5
    >>> vz = np.random.random(Npts)-0.5
    >>> velocities = np.vstack((vx,vy,vz)).T
    
    >>> rbins = np.logspace(-2,-1,10)
    >>> sigma_12 = radial_pvd_vs_r(coords, velocities, rbins, period=period)
    
    See also 
    ---------
    ref:`galaxy_catalog_analysis_tutorial7`
    """
    
    #process input arguments
    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads, max_sample_size,
        approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2,\
        period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs =\
        _pairwise_velocity_stats_process_args(*function_args)
    
    rbins = _process_radial_bins(rbins, period, PBCs)
    
    #calculate velocity difference scale
    std_v1 = np.sqrt(np.std(velocities1[0,:]))
    std_v2 = np.sqrt(np.std(velocities2[0,:]))
    
    #build the marks.
    shift1 = np.repeat(std_v1,len(sample1))
    shift2 = np.repeat(std_v2,len(sample2))
    marks1 = np.vstack((sample1.T, velocities1.T, shift1)).T
    marks2 = np.vstack((sample2.T, velocities2.T, shift2)).T
    
    
    def marked_pair_counts(sample1, sample2, rbins, period, num_threads,
        do_auto, do_cross, marks1, marks2,
        weight_func_id, _sample1_is_sample2, approx_cell1_size,approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """
        
        if do_auto is True:
            D1D1, S1S1, N1N1 = velocity_marked_npairs_3d(
                sample1, sample1, rbins,
                weights1=marks1, weights2=marks1,
                weight_func_id = weight_func_id,
                period=period, num_threads=num_threads,
                approx_cell1_size = approx_cell1_size,
                approx_cell2_size = approx_cell1_size)
            D1D1 = np.diff(D1D1)
            S1S1 = np.diff(S1S1)
            N1N1 = np.diff(N1N1)
        else:
            D1D1=None
            D2D2=None
            N1N1=None
            N2N2=None
            S1S1=None
            S2S2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
            S1S2 = S1S1
            S2S2 = S1S1
        else:
            if do_cross==True:
                D1D2, S1S2, N1N2 = velocity_marked_npairs_3d(
                    sample1, sample2, rbins,
                    weights1=marks1, weights2=marks2,
                    weight_func_id = weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell1_size,
                    approx_cell2_size = approx_cell2_size)
                D1D2 = np.diff(D1D2)
                S1S2 = np.diff(S1S2)
                N1N2 = np.diff(N1N2)
            else: 
                D1D2=None
                N1N2=None
                S1S2=None
            if do_auto is True:
                D2D2, S2S2, N2N2 = velocity_marked_npairs_3d(sample2, sample2, rbins,
                    weights1=marks2, weights2=marks2,
                    weight_func_id = weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell2_size,
                    approx_cell2_size = approx_cell2_size)
                D2D2 = np.diff(D2D2)
                S2S2 = np.diff(S2S2)
                N2N2 = np.diff(N2N2)
            else: 
                D2D2=None
                N2N2=None
    
        return D1D1, D1D2, D2D2, S1S1, S1S2, S2S2, N1N1, N1N2, N2N2
    
    weight_func_id = 12
    V1V1,V1V2,V2V2, S1S1, S1S2, S2S2, N1N1,N1N2,N2N2 = marked_pair_counts(
        sample1, sample2, rbins, period,
        num_threads, do_auto, do_cross,
        marks1, marks2, weight_func_id,
        _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size)
    
    
    def _shifted_std(N, sum_x, sum_x_sqr):
        """
        calculate the variance
        """
        variance = (sum_x_sqr - (sum_x * sum_x)/N)/(N - 1)
        return np.sqrt(variance)
    
    #return results
    if _sample1_is_sample2:
        sigma_11 = _shifted_std(N1N1,V1V1,S1S1)
        return np.where(np.isfinite(sigma_11), sigma_11, 0.)
    else:
        if (do_auto is True) & (do_cross is True): 
            sigma_11 = _shifted_std(N1N1,V1V1,S1S1)
            sigma_12 = _shifted_std(N1N2,V1V2,S1S2)
            sigma_22 = _shifted_std(N2N2,V2V2,S2S2)
            return (np.where(np.isfinite(sigma_11), sigma_11, 0.), 
            np.where(np.isfinite(sigma_12), sigma_12, 0.), 
            np.where(np.isfinite(sigma_22), sigma_22, 0.))
        elif (do_cross is True):
            sigma_12 = _shifted_std(N1N2,V1V2,S1S2)
            return np.where(np.isfinite(sigma_12), sigma_12, 0.)
        elif (do_auto is True):
            sigma_11 = _shifted_std(N1N1,V1V1,S1S1)
            sigma_22 = _shifted_std(N2N2,V2V2,S2S2)
            return (np.where(np.isfinite(sigma_11), sigma_11, 0.), 
                np.where(np.isfinite(sigma_22), sigma_22, 0.))


def mean_los_velocity_vs_rp(sample1, velocities1, rp_bins, pi_max,
                            sample2=None, velocities2=None,
                            period=None, do_auto=True, do_cross=True,
                            num_threads=1, max_sample_size=int(1e6),
                            approx_cell1_size = None,
                            approx_cell2_size = None):
    """ 
    Calculate the mean pairwise line-of-sight (LOS) velocity 
    as a function of projected seperation, :math:`\\bar{v}_{z,12}(r_p)`.
    
    Example calls to this function appear in the documentation below. 

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    velocities1 : array_like
        N1pts x 3 array containing the 3-D components of the velocities.
    
    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which 
        pairs are counted.
    
    pi_max : float
        maximum LOS seperation
    
    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.
        
    velocities2 : array_like, optional
        N2pts x 3 array containing the 3-D components of the velocities.
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].
    
    do_auto : boolean, optional
        caclulate the auto-pairwise velocities?
    
    do_cross : boolean, optional
        caclulate the cross-pairwise velocities?
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled
        such that the subsample is equal to max_sample_size.
    
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
        Analogous to ``approx_cell1_size``, but for `sample2`.  See comments for 
        ``approx_cell1_size`` for details. 
    
    Returns 
    -------
    vz_12 : numpy.array
        *len(rbins)-1* length array containing the mean pairwise LOS velocity, 
        :math:`\\bar{v}_{z12}(r)`, computed in each of the bins defined by ``rp_bins``.
    
    Notes
    -----
    The pairwise LOS velocity, :math:`v_{z12}(r)`, is defined as:
    
    .. math::
        v_{z12} = |\\vec{v}_{\\rm 1, pec} \\cdot \\hat{z}-\\vec{v}_{\\rm 2, pec}\\cdot\\hat{z}|
    
    where :math:`\\vec{v}_{\\rm 1, pec}` is the peculiar velocity of object 1, and 
    :math:`\\hat{z}` is the unit-z vector.
    
    :math:`\\bar{v}_{z12}(r_p)` is the mean of this quantity in projected radial bins.
    
    Pairs and radial velocities are calculated using 
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs_xy_z`.
    
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
    
    We will do the same to get a random set of peculiar velocities.
    
    >>> vx = np.random.random(Npts)-0.5
    >>> vy = np.random.random(Npts)-0.5
    >>> vz = np.random.random(Npts)-0.5
    >>> velocities = np.vstack((vx,vy,vz)).T
    
    >>> rp_bins = np.logspace(-2,-1,10)
    >>> pi_max = 0.3
    >>> vz_12 = mean_los_velocity_vs_rp(coords, velocities, rp_bins, pi_max, period=period)
    
    """
    
    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    
    sample1, velocities1, sample2, velocities2, period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs = _pairwise_velocity_stats_process_args(*function_args)
    
    rp_bins, pi_max = _process_rp_bins(rp_bins, pi_max, period, PBCs)
    pi_bins = np.array([0.0,pi_max])
    
    #create marks for the marked pair counter.
    marks1 = np.vstack((sample1.T, velocities1.T)).T
    marks2 = np.vstack((sample2.T, velocities2.T)).T
    
    def marked_pair_counts(sample1, sample2, rp_bins, pi_bins, period, num_threads,
        do_auto, do_cross, marks1, marks2,
        weight_func_id, _sample1_is_sample2, approx_cell1_size,approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """
        
        if do_auto is True:
            D1D1, dummy, N1N1 = velocity_marked_npairs_xy_z(
                sample1, sample1, rp_bins, pi_bins,
                weights1=marks1, weights2=marks1,
                weight_func_id = weight_func_id,
                period=period, num_threads=num_threads,
                approx_cell1_size = approx_cell1_size,
                approx_cell2_size = approx_cell1_size)
            D1D1 = np.diff(D1D1,axis=1)[:,0]
            D1D1 = np.diff(D1D1)
            N1N1 = np.diff(N1N1,axis=1)[:,0]
            N1N1 = np.diff(N1N1)
        else:
            D1D1=None
            D2D2=None
            N1N1=None
            N2N2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
        else:
            if do_cross is True:
                D1D2, dummy, N1N2 = velocity_marked_npairs_xy_z(
                    sample1, sample2, rp_bins, pi_bins,
                    weights1=marks1, weights2=marks2,
                    weight_func_id = weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell1_size,
                    approx_cell2_size = approx_cell2_size)
                D1D2 = np.diff(D1D2,axis=1)[:,0]
                D1D2 = np.diff(D1D2)
                N1N2 = np.diff(N1N2,axis=1)[:,0]
                N1N2 = np.diff(N1N2)
            else: 
                D1D2=None
                N1N2=None
            if do_auto is True:
                D2D2, dummy, N2N2 = velocity_marked_npairs_xy_z(
                    sample2, sample2, rp_bins, pi_bins,
                    weights1=marks2, weights2=marks2,
                    weight_func_id = weight_func_id, period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell2_size,
                    approx_cell2_size = approx_cell2_size)
                D2D2 = np.diff(D2D2,axis=1)[:,0]
                D2D2 = np.diff(D2D2)
                N2N2 = np.diff(N2N2,axis=1)[:,0]
                N2N2 = np.diff(N2N2)
            else: 
                D2D2=None
                N2N2=None
    
        return D1D1, D1D2, D2D2, N1N1, N1N2, N2N2
    
    
    #count the sum of radial velocities and number of pairs
    weight_func_id = 13
    V1V1,V1V2,V2V2, N1N1,N1N2,N2N2 =\
        marked_pair_counts(sample1, sample2, rp_bins, pi_bins, period,
            num_threads, do_auto, do_cross,
            marks1, marks2, weight_func_id,_sample1_is_sample2,
            approx_cell1_size, approx_cell2_size)
    
    #return results: the sum of radial velocities divided by the number of pairs
    if _sample1_is_sample2:
        M_11 = V1V1/N1N1
        return np.where(np.isfinite(M_11), M_11, 0.)
    else:
        if (do_auto is True) & (do_cross is True): 
            M_11 = V1V1/N1N1
            M_12 = V1V2/N1N2
            M_22 = V2V2/N2N2
            return (np.where(np.isfinite(M_11), M_11, 0.), 
                np.where(np.isfinite(M_12), M_12, 0.), 
                np.where(np.isfinite(M_22), M_22, 0.))
        elif (do_cross is True):
            M_12 = V1V2/N1N2
            return np.where(np.isfinite(M_12), M_12, 0.)
        elif (do_auto is True):
            M_11 = V1V1/N1N1
            M_22 = V2V2/N2N2 
            return np.where(np.isfinite(M_11), M_11, 0.), np.where(np.isfinite(M_22), M_22, 0.)


def los_pvd_vs_rp(sample1, velocities1, rp_bins, pi_max, sample2=None,
    velocities2=None, period=None, do_auto=True, do_cross=True,
    num_threads=1, max_sample_size=int(1e6),
    approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the pairwise line-of-sight (LOS) velocity dispersion (PVD), :math:`\\sigma_{z12}(r_p)`.
    
    Example calls to this function appear in the documentation below. 
    
    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    velocities1 : array_like
        N1pts x 3 array containing the 3-D components of the velocities.
    
    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which 
        pairs are counted.
    
    pi_max : float
        maximum LOS seperation
    
    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.
        
    velocities2 : array_like, optional
        N1pts x 3 array containing the 3-D components of the velocities.
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].
    
    do_auto : boolean, optional
        caclulate the auto-pairwise velocities?
    
    do_cross : boolean, optional
        caclulate the cross-pairwise velocities?
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled
        such that the subsample is equal to max_sample_size.
    
    Returns 
    -------
    sigma_12 : numpy.array
        *len(rbins)-1* length array containing the dispersion of the pairwise velocity, 
        :math:`\\sigma_{12}(r)`, computed in each of the bins defined by ``rbins``.
    
    Notes
    -----
    The pairwise LOS velocity, :math:`v_{z12}(r)`, is defined as:
    
    .. math::
        v_{z12} = |\\vec{v}_{\\rm 1, pec}\\cdot \\hat{z}-\\vec{v}_{\\rm 2, pec}\\cdot\\hat{z}|
    
    where :math:`\\vec{v}_{\\rm 1, pec}` is the peculiar velocity of object 1, and 
    :math:`\\hat{z}` is the unit-z vector.
    
    :math:`\\sigma_{z12}(r_p)` is the standard deviation of this quantity in 
    projected radial bins.
    
    Pairs and radial velocities are calculated using 
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs_xy_z`.
    
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
    
    We will do the same to get a random set of peculiar velocities.
    
    >>> vx = np.random.random(Npts)
    >>> vy = np.random.random(Npts)
    >>> vz = np.random.random(Npts)
    >>> velocities = np.vstack((vx,vy,vz)).T
    
    >>> rp_bins = np.logspace(-2,-1,10)
    >>> pi_max = 0.3
    >>> sigmaz_12 = los_pvd_vs_rp(coords, velocities, rp_bins, pi_max, period=period)

    >>> x2 = np.random.random(Npts)
    >>> y2 = np.random.random(Npts)
    >>> z2 = np.random.random(Npts)
    >>> coords2 = np.vstack((x2,y2,z2)).T

    >>> vx2 = np.random.random(Npts)
    >>> vy2 = np.random.random(Npts)
    >>> vz2 = np.random.random(Npts)
    >>> velocities2 = np.vstack((vx2,vy2,vz2)).T

    >>> sigmaz_12 = los_pvd_vs_rp(coords, velocities, rp_bins, pi_max, period=period, sample2=coords2, velocities2=velocities2)


    
    """
    
    #process input arguments
    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads, max_sample_size,
        approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2,\
        period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs =\
        _pairwise_velocity_stats_process_args(*function_args)
    
    rp_bins, pi_max = _process_rp_bins(rp_bins, pi_max, period, PBCs)
    pi_bins = np.array([0.0,pi_max])
    
    #calculate velocity difference scale
    std_v1 = np.sqrt(np.std(velocities1[2,:]))
    std_v2 = np.sqrt(np.std(velocities2[2,:]))
    
    #build the marks.
    shift1 = np.repeat(std_v1,len(sample1))
    shift2 = np.repeat(std_v2,len(sample2))
    marks1 = np.vstack((sample1.T, velocities1.T, shift1)).T
    marks2 = np.vstack((sample2.T, velocities2.T, shift2)).T

    
    def marked_pair_counts(sample1, sample2, rp_bins, pi_bins, period, num_threads,
        do_auto, do_cross, marks1, marks2,
        weight_func_id, _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """
        
        if do_auto is True:
            D1D1, S1S1, N1N1 = velocity_marked_npairs_xy_z(
                sample1, sample1, rp_bins, pi_bins,
                weights1=marks1, weights2=marks1, weight_func_id = weight_func_id, 
                period=period, num_threads=num_threads,
                approx_cell1_size = approx_cell1_size,
                approx_cell2_size = approx_cell1_size)
            D1D1 = np.diff(D1D1,axis=1)[:,0]
            D1D1 = np.diff(D1D1)
            S1S1 = np.diff(S1S1,axis=1)[:,0]
            S1S1 = np.diff(S1S1)
            N1N1 = np.diff(N1N1,axis=1)[:,0]
            N1N1 = np.diff(N1N1)
        else:
            D1D1=None
            D2D2=None
            N1N1=None
            N2N2=None
            S1S1=None
            S2S2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
            S1S2 = S1S1
            S2S2 = S1S1
        else:
            if do_cross is True:
                D1D2, S1S2, N1N2 = velocity_marked_npairs_xy_z(
                    sample1, sample2, rp_bins, pi_bins,
                    weights1=marks1, weights2=marks2, 
                    weight_func_id = weight_func_id, period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell1_size,
                    approx_cell2_size = approx_cell2_size)
                D1D2 = np.diff(D1D2,axis=1)[:,0]
                D1D2 = np.diff(D1D2)
                S1S2 = np.diff(S1S2,axis=1)[:,0]
                S1S2 = np.diff(S1S2)
                N1N2 = np.diff(N1N2,axis=1)[:,0]
                N1N2 = np.diff(N1N2)
            else: 
                D1D2=None
                N1N2=None
                S1S2=None
            if do_auto is True:
                D2D2, S2S2, N2N2 = velocity_marked_npairs_xy_z(
                    sample2, sample2, rp_bins, pi_bins,
                    weights1=marks2, weights2=marks2, 
                    weight_func_id = weight_func_id, period=period, num_threads=num_threads,
                    approx_cell1_size = approx_cell2_size,
                    approx_cell2_size = approx_cell2_size)
                D2D2 = np.diff(D2D2,axis=1)[:,0]
                D2D2 = np.diff(D2D2)
                S2S2 = np.diff(S2S2,axis=1)[:,0]
                S2S2 = np.diff(S2S2)
                N2N2 = np.diff(N2N2,axis=1)[:,0]
                N2N2 = np.diff(N2N2)
            else: 
                D2D2=None
                N2N2=None
    
        return D1D1, D1D2, D2D2, S1S1, S1S2, S2S2, N1N1, N1N2, N2N2
    
    weight_func_id = 14
    V1V1,V1V2,V2V2, S1S1, S1S2, S2S2, N1N1,N1N2,N2N2 = marked_pair_counts(
        sample1, sample2, rp_bins, pi_bins, period,
        num_threads, do_auto, do_cross, 
        marks1, marks2, weight_func_id, 
        _sample1_is_sample2, 
        approx_cell1_size, approx_cell2_size)
    
    def _shifted_std(N, sum_x, sum_x_sqr):
        """
        calculate the variance
        """
        variance = (sum_x_sqr - (sum_x * sum_x)/N)/(N - 1)
        return np.sqrt(variance)
    
    #return results
    if _sample1_is_sample2:
        sigma_11 = _shifted_std(N1N1,V1V1,S1S1)
        return np.where(np.isfinite(sigma_11), sigma_11, 0.)
    else:
        if (do_auto is True) & (do_cross is True): 
            sigma_11 = _shifted_std(N1N1,V1V1,S1S1)
            sigma_12 = _shifted_std(N1N2,V1V2,S1S2)
            sigma_22 = _shifted_std(N2N2,V2V2,S2S2)
            return (np.where(np.isfinite(sigma_11), sigma_11, 0.), 
                np.where(np.isfinite(sigma_12), sigma_12, 0.), 
                np.where(np.isfinite(sigma_22), sigma_22, 0.))
        elif (do_cross is True):
            sigma_12 = _shifted_std(N1N2,V1V2,S1S2)
            return np.where(np.isfinite(sigma_12), sigma_12, 0.)
        elif (do_auto is True):
            sigma_11 = _shifted_std(N1N1,V1V1,S1S1)
            sigma_22 = _shifted_std(N2N2,V2V2,S2S2)
            return (np.where(np.isfinite(sigma_11), sigma_11, 0.), 
                np.where(np.isfinite(sigma_22), sigma_22, 0.))


            

