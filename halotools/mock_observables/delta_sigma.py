# -*- coding: utf-8 -*-

"""
functions to calculate galaxy-galaxy lensing signal.
"""

from __future__ import division, print_function
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma
from .tpcf import tpcf
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from .clustering_helpers import *
from ..custom_exceptions import *
##########################################################################################

__all__=['delta_sigma']
__author__ = ['Duncan Campbell']


def delta_sigma(galaxies, particles, rp_bins, pi_max, period=None, log_bins=True,\
                n_bins=25, estimator='Natural', num_threads=1):
    """ 
    Calculate the galaxy-galaxy lensing signal :math:`\\Delta\\Sigma(r_p)`.
    
    This function computes the cross correlation between `galaxies` and `particles`
    to get :math:`\\xi_{\\rm galaxies, matter}(r)`, and integrates the result to 
    get :math:`\\Delta\\Sigma(r_p)`.
    
    Parameters
    ----------
    galaxies : array_like
        Ngal x 3 numpy array containing 3-d positions of galaxies.
    
    particles : array_like
        Npart x 3 numpy array containing 3-d positions of partciles.
    
    rp_bins : array_like
        array of projected radial boundaries defining the bins in which the result is 
        calculated.  The minimum of rp_bins must be > 0.0.
    
    pi_max: float
        maximum integration parameter, :math:`\\pi_{\\rm max}` (see notes).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    log_bins : boolean, optional
        integration parameter (see notes)
    
    n_bins : int, optional
        integration parameter (see notes)
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
        
    Returns 
    -------
    Delta_Sigma: np.array
        :math:`\\Delta\\Sigma(r_p)` calculated in projected radial bins defined by 
        rp_bins.  The units are units(particles)/units(rp_bins)**2.
    
    Notes
    -----
    :math:`\\Delta\\Sigma` is calculated by first calculating,
    :math:`\\Sigma(r_p) = \\bar{\\rho}\\int_0^{\\pi_{\\rm max}} \\left[1+\\xi_{\\rm g,m}(\\sqrt{r_p^2+\\pi^2}) \\right]\\mathrm{d}\\pi`
    and then,
    :math:`\\Delta\\Sigma(r_p) = \\bar{\\Sigma}(<r_p) - \\Sigma(r_p)`
    where,
    :math:`\\bar{\\Sigma}(<r_p) = \\frac{1}{\\pi r_p^2}\\int_0^{r_p}\\Sigma(r_p^{\\prime})2\\pi r_p^{\\prime} \\mathrm{d}r_p^{\\prime}`
    
    Numerically,
    :math:`\\xi(r)` is calculated in `n_bins` evenly spaced linearly or log-linearly as
    indicated by `log_bins`.
    
    All integrals are done use scipy.integrate.quad.
    """
    
    #process the input parameters
    function_args = [galaxies, particles, rp_bins, pi_max, period, estimator, num_threads]
    galaxies, particles, rp_bins, period, num_threads, PBCs =\
        _delta_sigma_process_args(*function_args)
    
    mean_rho = len(particles)/period.prod() #number density of particles
    
    #determine radial bins to calculate tpcf in
    rp_max = np.max(rp_bins)
    rp_min = np.min(rp_bins)
    #maximum radial distance to calculate TPCF out to:
    rmax = np.sqrt(rp_max**2 + pi_max**2)
    
    #check to make sure rmax is not too large
    if (period is not None):
        if (rmax >= np.min(period)/3.0):
            msg = ("\n"
                   "rmax = sqrt(max(rp_bins)**2 + pi_max**2)>Lbox/3 \n"
                   "The DeltaSigma calculation requires the correlation function \n"
                   "to be calculated out to rmax. The maximum length over which you \n"
                   "search for pairs of points cannot be larger than Lbox/3 \n"
                   "in any dimension. If you need to count pairs on these \n"
                   "length scales, you should use a larger simulation.")
            raise HalotoolsError(msg)
    
    #define radial bins using either log or linear spacing
    if log_bins==True:
        rbins = np.logspace(np.log10(rp_min), np.log10(rmax), n_bins)
    else: 
        rbins = np.linspace(rp_min, rmax, n_bins)
    
    #calculate the cross-correlation between galaxies and particles
    xi = tpcf(galaxies, rbins, sample2=particles, randoms=None, period=period,\
              do_auto=False, do_cross=True, estimator=estimator, num_threads=num_threads)
    
    #fit a spline to the tpcf
    #note that we fit the log10 of xi+1.0
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0 #note these are the true centers, not log
    xi = InterpolatedUnivariateSpline(rbin_centers, np.log10(xi+1.0), ext=0)
    
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


