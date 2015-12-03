# -*- coding: utf-8 -*-

"""
Calculate the multipoles of the two point correlation function
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from scipy.special import legendre
##########################################################################################

__all__ = ['multipole']

__author__ = ['Duncan Campbell']

def multipole(x, mu_bins, order=0):
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
    
    """
    mu_bin_centers = (mu_bins[:-1]+mu_bins[1:])/(2.0)

    Ln = legendre(order)
    result = (2.0*order + 1.0)/2.0 * np.sum(x*np.diff(mu_bins)*Ln(mu_bin_centers),axis=1)
    return result
