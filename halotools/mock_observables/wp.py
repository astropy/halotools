# -*- coding: utf-8 -*-

"""
functions to calculate clustering statistics, e.g. two point correlation functions.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
####import modules########################################################################
import numpy as np

from .clustering_helpers import _rp_pi_tpcf_process_args
from .rp_pi_tpcf import rp_pi_tpcf
##########################################################################################


__all__=['wp']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def wp(sample1, rp_bins, pi_max, sample2=None, randoms=None, period=None,
    do_auto=True, do_cross=True, estimator='Natural', num_threads=1,
    max_sample_size=int(1e6), approx_cell1_size=None, approx_cell2_size=None,
    approx_cellran_size=None):
    """ 
    Calculate the projected two point correlation function, :math:`w_{p}(r_p)`,
    where :math:`r_p` is the seperation perpendicular to the line-of-sight (LOS).
    
    Calculation of :math:`w_{p}(r_p)` requires the user to supply bins in :math:`\\pi`,
    the seperation parallel to the line of sight, and the result will in general depend 
    on both the binning and the maximum :math:`\\pi` seperation integrated over.  See 
    notes for further details.
    
    The first two dimensions define the plane for perpendicular distances.  The third 
    dimension is used for parallel distances.  i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate. This is the 'distant observer' approximation.
    
    Example calls to this function appear in the documentation below. 
    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` and ``sample2`` arguments. 
      
    See also :ref:`galaxy_catalog_analysis_tutorial4`. 

    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points. 
    
    rp_bins : array_like
        array of boundaries defining the bins perpendicular to the LOS in which 
        pairs are counted.
    
    pi_max : float
        maximum LOS distance to to search for pairs when calculating math:`w_p`.
        see Notes.
    
    sample2 : array_like, optional
        Npts x 3 numpy array containing 3-D positions of points.
    
    randoms : array_like, optional
        Nran x 3 numpy array containing 3-D positions of points.
    
    period : array_like, optional
        Length-k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be [Lbox]*3.
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
        
        If sample size exceeds max_sample_size, the sample will be randomly down-sampled 
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
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details. 
    
    approx_cellran_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for randoms.  See comments for 
        ``approx_cell1_size`` for details. 

    Returns 
    -------
    correlation_function(s) : numpy.array
        *len(rp_bins)-1* length array containing the correlation function :math:`w_p(r_p)` 
        computed in each of the bins defined by input ``rp_bins``.
        
        
        If ``sample2`` is not None (and not exactly the same as ``sample1``), 
        three arrays of length *len(rp_bins)-1* are returned: 
        
        .. math::
            w_{p11}(r_p), \\ w_{p12}(r_p), \\ w_{p22}(r_p),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` 
        and ``sample2``, and the autocorrelation of ``sample2``.  If ``do_auto`` or ``do_cross`` 
        is set to False, the appropriate result(s) is not returned.
    
    Notes
    -----
    The projected correlation function is calculated by integrating the 
    redshift space two point correlation function using 
    `~halotools.mock_observables.rp_pi_tpcf`:
    
    .. math::
        w_p(r_p) = \\int_0^{\\pi_{\\rm max}}\\xi(r_p,\\pi)\\mathrm{d}\\pi
    
    where :math:`\\pi_{\\rm max}` is ``pi_max`` and :math:`\\xi(r_p,\\pi)` 
    is the redshift space correlation function.

    For a higher-performance implementation of the wp function, 
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
    
    >>> rp_bins = np.logspace(-2,-1,10)
    >>> pi_max = 0.1
    >>> xi = wp(coords, rp_bins, pi_max, period=period)
    
    See also 
    --------
    :ref:`galaxy_catalog_analysis_tutorial4`

    """
    
    #define the volume to search for pairs
    pi_max = float(pi_max)
    pi_bins = np.array([0.0,pi_max])
    
    #process input parameters
    function_args = (sample1, rp_bins, pi_bins, sample2, randoms, period, do_auto,
        do_cross, estimator, num_threads, max_sample_size,
        approx_cell1_size, approx_cell2_size, approx_cellran_size)
    sample1, rp_bins, pi_bins, sample2, randoms, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = _rp_pi_tpcf_process_args(*function_args)
    
    if _sample1_is_sample2:
        sample2=None
    
    #pass the arguments into the redshift space TPCF function
    result = rp_pi_tpcf(sample1, rp_bins=rp_bins, pi_bins=pi_bins,
        sample2 = sample2, randoms=randoms,
        period = period, do_auto=do_auto, do_cross=do_cross,
        estimator=estimator, num_threads=num_threads,
        max_sample_size=max_sample_size,
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell2_size,
        approx_cellran_size=approx_cellran_size)
    
    #return the results.
    if _sample1_is_sample2:
        D1D1 = result[:,0]
        wp_D1D1 = 2.0*D1D1*pi_max
        return wp_D1D1
    else:
        if (do_auto is True) & (do_cross is True):
            D1D1 = result[0][:,0]
            D1D2 = result[1][:,0]
            D2D2 = result[2][:,0]
            wp_D1D1 = 2.0*D1D1*pi_max
            wp_D1D2 = 2.0*D1D2*pi_max
            wp_D2D2 = 2.0*D2D2*pi_max
            return wp_D1D1, wp_D1D2, wp_D2D2
        elif (do_auto is True) & (do_cross is False):
            D1D1 = result[0][:,0]
            D2D2 = result[1][:,0]
            wp_D1D1 = 2.0*D1D1*pi_max
            wp_D2D2 = 2.0*D2D2*pi_max
            return wp_D1D1, wp_D2D2
        elif (do_auto is False) & (do_cross is True):
            D1D2 = result[:,0]
            wp_D1D2 = 2.0*D1D2*pi_max
            return wp_D1D2


