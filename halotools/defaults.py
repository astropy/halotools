"""
This module expresses the default values for the halo occupation models.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)



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


### Default values specifying traditional quenching model
default_quenching_parameters = {
    'logM_abcissa' : [12,13.5,15],
    'central_ordinates' : [0.35,0.75,0.95], #polynomial coefficients determining quenched fraction of centrals
    'satellite_ordinates' : [0.5,0.75,0.85] #polynomial coefficients determining quenched fraction of centrals
    }

