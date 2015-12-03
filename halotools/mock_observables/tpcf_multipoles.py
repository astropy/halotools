# -*- coding: utf-8 -*-

"""
Calculate the multipoles of the two point correlation function
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from ..utils.array_utils import convert_to_ndarray
from ..custom_exceptions import *
from warnings import warn
from scipy.special import legendre
##########################################################################################

__all__ = ['tpcf_multipole']

__author__ = ['Duncan Campbell']

def tpcf_multipole(s_mu_tcpf_result, mu_bins, order=0):
    """
    Calculate the multipoles of the two point correlation funcion.
    
    Parameters
    ----------
    s_mu_tcpf_result : np.ndarray
        2-D array with the two point correlation function calculated in bins 
        of :math:`s` and :math:`\\mu`.  See `~halotools.mock_observables.s_mu_tpcf`.
    
    mu_bins : array_like
        array of :math:`\\mu` bins for which ``s_mu_tcpf_result`` has been calculated.
    
    order : int, optional
        order of the multpole returned.
    
    Returns:
    -------
    xi_l : np.array
        multipole of ``s_mu_tcpf_result`` of the indicated order.
    
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
    
    First, we calculate the correlation function using 
    `~halotools.mock_observables.s_mu_tpcf`.
    
    >>> from halotools.mock_observables import s_mu_tpcf
    >>> s_bins  = np.linspace(0,0.25,10)
    >>> mu_bins = np.linspace(0,1,100)
    >>> xi = s_mu_tpcf(coords, s_bins, mu_bins, period=period)
    
    Then, we can claclulate the quadrapole of the correlatio function:
    
    >>> xi_2 = multipole(xi, mu_bins, order=2)
    """
    
    #process inputs
    s_mu_tcpf_result = convert_to_ndarray(s_mu_tcpf_result)
    mu_bins = convert_to_ndarray(mu_bins)
    order = int(order)
    
    #calculate the center of each mu bin
    mu_bin_centers = (mu_bins[:-1]+mu_bins[1:])/(2.0)
    
    #get the Legendre polynomial of the desired order.
    Ln = legendre(order)
    
    #numerically integrate over mu
    result = (2.0*order + 1.0)/2.0 *\
             np.sum(s_mu_tcpf_result * np.diff(mu_bins) * Ln(mu_bin_centers), axis=1)
    
    return result
