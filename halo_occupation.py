# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:52:05 2014

@author: aphearin
"""

import numpy as np
from scipy.special import erf
from scipy.stats import poisson

def mean_ncen(logM,hod_dict=None):
    """Expected number of central galaxies in a halo of mass 10**logM"""

    if hod_dict == None:
    	hod_dict['logMmin_cen'] = 11.75
    	hod_dict['sigma_logM'] = 0.2

    mean_ncen = 0.5*(1.0 + erf((logM - hod_dict['logMmin_cen'])/hod_dict['sigma_logM']))
    return mean_ncen


def num_ncen(logM,hod_dict):
	"""Returns 1 or 0 for whether or not there is a central in this halo."""
	return np.array(mean_ncen(logM,hod_dict) > np.random.random(len(logM)),dtype=int)


def mean_nsat(logM,hod_dict=None):
    """Expected number of satellite galaxies in a halo of mass 10**logM."""
    halo_mass = 10.**logM

    if hod_dict == None:
    	hod_dict['logMmin_sat'] = 12.25
    	hod_dict['Msat_ratio'] = 20.0
    	hod_dict['alpha_sat'] = 1.0

    Mmin_sat = 10.**hod_dict['logMmin_sat']
    M1_sat = hod_dict['Msat_ratio']*Mmin_sat

    mean_nsat = np.zeros(len(logM),dtype='f8')
    idx_nonzero_satellites = (halo_mass - Mmin_sat) > 0
    mean_nsat[idx_nonzero_satellites] = ((halo_mass[idx_nonzero_satellites] - Mmin_sat)/M1_sat)**hod_dict['alpha_sat']

    return mean_nsat

def num_nsat(logM,hod_dict):
	'''    Returns a random number of satellites in a halo.'''
	Prob_sat = mean_nsat(logM,hod_dict)
	# NOTE: need to cut at zero, otherwise poisson bails
    # BUG IN SCIPY: poisson.rvs bails if there are zeroes in a numpy array
	test = Prob_sat <= 0
	Prob_sat[test] = 1.e-20

	return poisson.rvs(Prob_sat)







