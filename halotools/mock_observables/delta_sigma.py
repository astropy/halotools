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
##########################################################################################

__all__=['delta_sigma']
__author__ = ['Duncan Campbell']


def delta_sigma(galaxies, particles, rp_bins, pi_max, period=None, log_bins=True,\
                n_bins=25, estimator='Natural', num_threads=1):
    """ 
    Calculate the galaxy-galaxy lensing signal :math:`\\Delta\\Sigma(r_p)`.
    
    Parameters
    ----------
    galaxies : array_like
        Ngal x 3 numpy array containing 3-d positions of galaxies.
    
    particles : array_like
        Npart x 3 numpy array containing 3-d positions of partciles.
    
    rp_bins : array_like
        array of projected radial boundaries defining the bins in which the result is 
        calculated.
    
    pi_max: float
        maximum integration parameter, :math:`\\pi_{\\rm max}` (see notes).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    log_bins : boolean, optional
        integration parameter
    
    n_bins : int, optional
        integration parameter
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
        
    Returns 
    -------
    Delta_Sigma: np.array
        :math:`\\Delta\\Sigma(r_p)` calculated in projected radial bins defined by 
        rp_bins.
    
    Notes
    -----
    :math:`\\Delta\\Sigma` is calculated by first calculating,
    :math:`\\Sigma(r_p) = \\bar{\\rho}\\int_0^{\\pi_{\\rm max}} \\left[1+\\xi_{\\rm g,m}(\\sqrt{r_p^2+\\pi^2}) \\right]\\mathrm{d}\\pi`
    and then,
    :math:`\\Delta\\Sigma(r_p) = \\bar{\\Sigma}(<r_p) - \\Sigma(r_p)`
    """
    
    #process the input parameters
    galaxies, particles, rp_bins, period, num_threads, PBCs = \
        _delta_sigma_process_args(galaxies, particles, rp_bins, pi_max, period, estimator, num_threads)
    
    mean_rho = len(particles)/period.prod()
    
    #determine radial bins to calculate tpcf in
    rp_max = np.max(rp_bins)
    rp_min = np.min(rp_bins)
    rmax = np.sqrt(rp_max**2 + pi_max**2)
    
    if log_bins==True:
        rbins = np.logspace(np.log10(rp_min), np.log10(rmax), n_bins)
    else: 
        rbins = np.linspace(rp_min, rmax, n_bins)
    
    #calculate the cross-correlation between galaxies and particles
    xi = tpcf(galaxies, rbins, sample2=particles, randoms=None, period=period,\
              do_auto=False, do_cross=True, estimator=estimator, num_threads=num_threads)
    
    #fit a spline to the tpcf
    #note that we fit the log of xi
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0
    rmax = np.max(rbin_centers)
    xi = InterpolatedUnivariateSpline(np.log10(rbin_centers), np.log10(xi), ext=0)
    
    #define function to integrate
    def f(pi,rp):
        x = np.sqrt(rp**2+pi**2)
        return mean_rho*(1.0+10.0**xi(np.log10(x)))
    
    #integrate xi to get the surface density as a function of r_p
    sigma = np.zeros(len(rp_bins))
    for i in range(0,len(rp_bins)):
        sigma[i] = integrate.quad(f,0.0,pi_max,args=(rp_bins[i],))[0]
    
    #integrate sigma to the mean internal surface density
    sigma = InterpolatedUnivariateSpline(rp_bins, sigma, ext=0)
    def f(rp):
        return sigma(rp)*2.0*np.pi*rp
    
    mean_internal_sigma = np.zeros(len(rp_bins))
    for i in range(0,len(rp_bins)):
        mean_internal_sigma[i] = integrate.quad(f,0.0,rp_bins[i])[0]/(np.pi*rp_bins[i]**2.0)
        
    delta_sigma = mean_internal_sigma - sigma(rp_bins)
    
    return delta_sigma
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    