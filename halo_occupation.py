# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:52:05 2014

@author: aphearin
"""

import numpy as np
from scipy.special import erf

def mean_ncen(logM,hod_dict=None):
    """Expected number of central galaxies in a halo of mass 10**logM"""

    if hod_dict == None:
    	hod_dict['logMmin_cen'] = 11.75
    	hod_dict['sigma_logM'] = 0.2

    mean_ncen = 0.5*(1.0 + erf((logM - hod_dict['logMmin_cen'])/hod_dict['sigma_logM']))
    return mean_ncen

#def ng_ncen(logM)

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








