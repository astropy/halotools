"""
This module expresses the default values for the halo occupation models.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import warnings
import numpy as np


class simulation(object):
    ''' Container class for properties of the simulation being used.
    
    Still unused.
    
    
    '''
    
    def __init__(self,simulation_nickname=None):
        
        if simulation_nickname is None:
            self.halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
            self.simulation_dict = {
            'catalog_filename':default_halo_catalog_filename,
            'Lbox':250.0,
            'scale_factor':1.0003,
            'particle_mass':1.35e8,
            'softening':1.0
            }
        elif simulation_nickname is 'Bolshoi':
            self.halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
            self.simulation_dict = {
            'catalog_filename':default_halo_catalog_filename,
            'Lbox':250.0,
            'scale_factor':1.0003,
            'particle_mass':1.35e8,
            'softening':1.0
            }
        else:
            pass
        
                    

### Default halo catalog
default_halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
default_simulation_dict = {
	'catalog_filename':default_halo_catalog_filename,
	'Lbox':250.0,
	'scale_factor':1.0003,
	'particle_mass':1.35e8,
	'softening':1.0
}


default_luminosity_threshold = -19.5
default_tiny_poisson_fluctuation = 1.e-20

default_Npts_concentration_array = 1000
default_Npts_radius_array = 101




### Default HOD model
default_hod_dict = {
	'logMmin_cen' : 11.6, # log Mass where < Ncen (M) > = 0.5
	'sigma_logM' : 0.2, # scatter in central galaxy stellar-to-halo mass
	'logMmin_sat' : 11.7, # low-end cutoff in log Mass for a halo to contain a satellite
	'Msat_ratio' : 20.0, # multiplicative factor specifying when a halo contains a satellite
	'alpha_sat' : 1.05, # power law slope of the satellite occupation function
    'fconc' : 0.5 # multiplicative factor used to scale satellite concentrations
}

default_color_dict = {
     'central_coefficients' : [0.35,0.75,0.95], #polynomial coefficients determining quenched fraction of centrals
     'satellite_coefficients' : [0.5,0.75,0.85], #polynomial coefficients determining quenched fraction of centrals
     'abcissa' : [12,13.5,15]
}


default_NFW_concentration_precision = 0.5











