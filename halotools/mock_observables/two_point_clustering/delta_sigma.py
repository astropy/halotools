"""
Module containing the `~halotools.mock_observables.delta_sigma` function used to 
calculate galaxy-galaxy lensing. 
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from warnings import warn

from .tpcf import tpcf

from ..mock_observables_helpers import (get_num_threads, get_separation_bins_array, 
    get_period, enforce_sample_respects_pbcs, enforce_sample_has_correct_shape)
from .clustering_helpers import verify_tpcf_estimator

__all__ = ['delta_sigma']
__author__ = ['Duncan Campbell']


def delta_sigma(galaxies, particles, rp_bins, pi_max, period,
    log_bins=True, n_bins=25, estimator='Natural', num_threads=1,
    approx_cell1_size = None, approx_cell2_size = None):
    """ 
    Calculate the galaxy-galaxy lensing signal :math:`\\Delta\\Sigma(r_p)`.
    
    This function computes the cross correlation between ``galaxies`` and ``particles``
    to get the galaxy-matter cross correlation, :math:`\\xi_{\\rm g, m}(r)`, and 
    integrates the result to get :math:`\\Delta\\Sigma(r_p)`.  See the notes for details 
    about the calculation.
    
    Example calls to this function appear in the documentation below. 
    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``galaxies`` and ``particles`` arguments.   

    See also :ref:`galaxy_catalog_analysis_tutorial3`. 

    Parameters
    ----------
    galaxies : array_like
        Ngal x 3 numpy array containing 3-d positions of galaxies.
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    particles : array_like
        Npart x 3 numpy array containing 3-d positions of particles.
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    rp_bins : array_like
        array of projected radial boundaries defining the bins in which the result is 
        calculated.  The minimum of rp_bins must be > 0.0.
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    pi_max: float
        maximum integration parameter, :math:`\\pi_{\\rm max}` 
        (see notes for more details).
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    period : array_like
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    log_bins : boolean, optional
        integration parameter (see notes for more details).
    
    n_bins : int, optional
        integration parameter (see notes for more details).
    
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
    
    Returns 
    -------
    Delta_Sigma : np.array
        :math:`\\Delta\\Sigma(r_p)` calculated at projected radial distances ``rp_bins``.
        The units are units(particles)/units(rp_bins)**2.
    
    Notes
    -----
    :math:`\\Delta\\Sigma` is calculated by first calculating,
    
    .. math::
        \\Sigma(r_p) = \\bar{\\rho}\\int_0^{\\pi_{\\rm max}} \\left[1+\\xi_{\\rm g,m}(\\sqrt{r_p^2+\\pi^2}) \\right]\\mathrm{d}\\pi
    
    and then,
    
    .. math::
        \\Delta\\Sigma(r_p) = \\bar{\\Sigma}(<r_p) - \\Sigma(r_p)
    
    where,
    
    .. math::
        \\bar{\\Sigma}(<r_p) = \\frac{1}{\\pi r_p^2}\\int_0^{r_p}\\Sigma(r_p^{\\prime})2\\pi r_p^{\\prime} \\mathrm{d}r_p^{\\prime}
    
    Numerically, :math:`\\xi_{\\rm g,m}` is calculated in `n_bins` evenly spaced linearly 
    or log-linearly as indicated by `log_bins` and integrated between 
    :math:`{\\rm rp}_{\\rm min}` and 
    :math:`\\sqrt{{\\rm{rp}_{\\rm max}}^2 + {\\pi_{\\rm max}}^2}`.
    
    All integrals are done use `scipy.integrate.quad`.
    
    Examples
    --------
    For demonstration purposes we will calculate `delta_sigma` using the 
    `~halotools.sim_manager.FakeSim` that is generated on-the-fly. 

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    
    
    >>> x = halocat.halo_table['halo_x']
    >>> y = halocat.halo_table['halo_y']
    >>> z = halocat.halo_table['halo_z']
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> galaxies = np.vstack((x,y,z)).T
    
    Let's do the same thing for a set of particle data
    
    >>> px = halocat.ptcl_table['x']
    >>> py = halocat.ptcl_table['y']
    >>> pz = halocat.ptcl_table['z']
    >>> particles = np.vstack((px,py,pz)).T
    
    >>> rp_bins = np.logspace(-1, 1, 10)
    >>> result = delta_sigma(galaxies, particles, rp_bins, pi_max=20, period=halocat.Lbox)

    See also 
    --------
    :ref:`galaxy_catalog_analysis_tutorial3`

    """

    #process the input parameters
    args = (galaxies, particles, rp_bins, period, estimator, num_threads)
    result = _delta_sigma_process_args(*args)
    galaxies, particles, rp_bins, period, estimator, num_threads, PBCs = result
    
    #determine radial bins to calculate tpcf in
    rp_max = np.max(rp_bins)
    rp_min = np.min(rp_bins)
    #maximum radial distance to calculate TPCF out to:
    rmax = np.sqrt(rp_max**2 + pi_max**2)
        
    #define radial bins using either log or linear spacing
    if log_bins is True:
        rbins = np.logspace(np.log10(rp_min), np.log10(rmax), n_bins)
    else: 
        rbins = np.linspace(rp_min, rmax, n_bins)
    
    #calculate the cross-correlation between galaxies and particles
    xi = tpcf(galaxies, rbins, sample2=particles, randoms=None, period=period,
        do_auto=False, do_cross=True, estimator=estimator, num_threads=num_threads,
        approx_cell1_size = approx_cell1_size, approx_cell2_size = approx_cell2_size)
    
    #Check to see if xi ever is equal to -1
    #if so, there are radial bins with 0 matter particles.
    #This could mean that the user has under-sampled the particles.
    if np.any(xi==-1.0):
        msg = ("\n"
               "Some radial bins contain 0 particles in the \n"
               "galaxy-matter cross cross correlation calculation. \n"
               "If you downsampled the amount of matter particles, \n"
               "consider using more particles. Alternatively, you \n"
               "may (also) want to use fewer bins in the calculation. \n"
               "Set the `n_bins` parameter to a smaller number. \n"
               "Finally, you may want to calculate Delta_Sigma to \n"
               "a larger minimum projected distance, min(`rp_bins`)."
               )
        warn(msg)
    
    
    #fit a spline to the tpcf
    #note that we fit the log10 of xi+1.0
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0 #note these are the true centers, not log
    xi = InterpolatedUnivariateSpline(rbin_centers, np.log10(xi+1.0), ext=0)
    
    mean_rho = len(particles)/period.prod() #number density of particles

    #define function to integrate
    def f(pi,rp):
        r = np.sqrt(rp**2+pi**2)
        #note that we take 10**xi-1,
        #because we fit the log xi
        return mean_rho*(1.0+(10.0**xi(r)-1.0))
    
    #integrate xi to get the surface density as a function of r_p
    surface_density = np.zeros(len(rp_bins)) #initialize to 0.0
    for i in range(0,len(rp_bins)):
        surface_density[i] = integrate.quad(f,0.0,pi_max,args=(rp_bins[i],))[0]
    
    #fit a spline to the surface density
    surface_density = InterpolatedUnivariateSpline(rp_bins, np.log10(surface_density), ext=0)
    
    #integrate surface density to get the mean internal surface density
    #define function to integrate
    def f(rp):
        #note that we take 10**surface_density,
        #because we fit the log of surface density
        return 10.0**surface_density(rp)*2.0*np.pi*rp
    
    #do integral to get mean internal surface density
    mean_internal_surface_density = np.zeros(len(rp_bins))
    for i in range(0,len(rp_bins)):
        internal_area = np.pi*rp_bins[i]**2.0
        mean_internal_surface_density[i] = integrate.quad(f,0.0,rp_bins[i])[0]/(internal_area)
    
    #calculate an return the change in surface density, delta sigma
    delta_sigma = mean_internal_surface_density - 10**surface_density(rp_bins)
    
    return delta_sigma

def _delta_sigma_process_args(galaxies, particles, rp_bins, period, estimator, num_threads):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.delta_sigma`. 
    """
    period, PBCs = get_period(period)

    galaxies = enforce_sample_has_correct_shape(galaxies)
    particles = enforce_sample_has_correct_shape(particles)
    enforce_sample_respects_pbcs(galaxies[:,0], galaxies[:,1], galaxies[:,2], period)
    enforce_sample_respects_pbcs(particles[:,0], particles[:,1], particles[:,2], period)

    rp_bins = get_separation_bins_array(rp_bins)

    num_threads = get_num_threads(num_threads, enforce_max_cores = False)
    
    estimator = verify_tpcf_estimator(estimator)
    
    return galaxies, particles, rp_bins, period, estimator, num_threads, PBCs

