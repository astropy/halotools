# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:52:05 2014

@author: aphearin
"""

import numpy as np
from scipy.special import erf
from scipy.stats import poisson
import defaults

def mean_ncen(logM,hod_dict=None):
    """ Expected number of central galaxies in a halo of mass 10**logM.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary
    Contains model parameters used to define the central occupation function.

    Returns
    ----------
    mean_ncen : float or array
    Mean number of central galaxies in a host halo of the specified mass. 
    Values are restricted 0 <= mean_ncen <= 1.

    Notes 
    ----------
    Math equation should be inserted here.


    """

    if hod_dict == None:
        hod_dict = defaults.default_hod_dict

    mean_ncen = 0.5*(1.0 + erf((logM - hod_dict['logMmin_cen'])/hod_dict['sigma_logM']))
    return mean_ncen


def num_ncen(logM,hod_dict):
    """ Returns Monte Carlo-generated array of 0 or 1 specifying whether there is a central in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary
    Contains model parameters used to define the central occupation function.

    Returns
    ----------
    num_ncen_array : int or array
    Array is unity if input halo hosts a central galaxy, 0 if not.

    Notes 
    ----------
    Should say something about the analytical function used to generate the Monte Carlo.
    """

    num_ncen_array = np.array(mean_ncen(logM,hod_dict) > np.random.random(len(logM)),dtype=int)
    return num_ncen_array

def mean_nsat(logM,hod_dict=None):
    """Expected number of satellite galaxies in a halo of mass 10**logM.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary
    Contains model parameters used to define the central occupation function.

    Returns
    ----------
    mean_nsat : float or array
    Mean number of satellite galaxies in a host halo of the specified mass. 

    Notes 
    ----------
    Math equation should be inserted here.

    """
    halo_mass = 10.**logM

    if hod_dict == None:
        hod_dict = defaults.default_hod_dict

    Mmin_sat = 10.**hod_dict['logMmin_sat']
    M1_sat = hod_dict['Msat_ratio']*Mmin_sat

    mean_nsat = np.zeros(len(logM),dtype='f8')
    idx_nonzero_satellites = (halo_mass - Mmin_sat) > 0
    mean_nsat[idx_nonzero_satellites] = ((halo_mass[idx_nonzero_satellites] - Mmin_sat)/M1_sat)**hod_dict['alpha_sat']

    return mean_nsat

def num_nsat(logM,hod_dict):
    '''  Returns Monte Carlo-generated array of integers specifying the number of satellites in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary
    Contains model parameters used to define the central occupation function.

    Returns
    ----------
    num_nsat_array : int or array
    Values of array specify the number of satellites hosted by each halo.

    Notes 
    ----------
    Should say something about the analytical function used to generate the Monte Carlo.

    '''
    Prob_sat = mean_nsat(logM,hod_dict)
	# NOTE: need to cut at zero, otherwise poisson bails
    # BUG IN SCIPY: poisson.rvs bails if there are zeroes in a numpy array
    test = Prob_sat <= 0
    Prob_sat[test] = 1.e-20

    num_nsat_array = poisson.rvs(Prob_sat)

    return num_nsat_array







