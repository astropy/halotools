# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:52:05 2014

@author: aphearin
"""

import numpy as np
from scipy.special import erf

def ncen(logM,logMmin_cen=11.75,sigma_logM=0.2):
    """Expected number of central galaxies in a halo of mass 10**logM"""
    mean_ncen = 0.5*(1.0 + erf((logM - logMmin_cen)/sigma_logM))
    return mean_ncen

def nsat(logM,logMmin_sat=12,Msat_ratio=20,alpha_sat=1):
    """Expected number of satellite galaxies in a halo of mass 10**logM."""
    halo_mass = 10.**logM
    Mmin_sat = 10.**logMmin_sat
    M1_sat = Msat_ratio*Mmin_sat

    mean_nsat = np.zeros(len(logM),dtype='f8')
    idx_nonzero_satellites = (halo_mass - Mmin_sat) > 0
    mean_nsat[idx_nonzero_satellites] = ((halo_mass[idx_nonzero_satellites] - Mmin_sat)/M1_sat)**alpha_sat

    return mean_nsat

    






