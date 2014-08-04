"""
Module expressing various default values of the mock-making code.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)



### Default halo catalog
default_halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/presorted_host_halo_catalog.fits'
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
    'logM_quenching_abcissa' : [12,13.5,15],
    'central_quenching_ordinates' : [0.35,0.75,0.95], #polynomial coefficients determining quenched fraction of centrals
    'satellite_quenching_ordinates' : [0.5,0.75,0.85] #polynomial coefficients determining quenched fraction of centrals
    }

default_assembias_parameters = {
	'logM_assembias_abcissa' : [12,14],
	'satellite_destruction_quenched_central_ordinates' : [1.1,1],
	'satellite_destruction_no_central_ordinates' : [1.0,1.0],
	'satellite_conformity_quenched_central_ordinates' : [1.1,1],
	'satellite_conformity_no_central_ordinates' : [1.0,1.0],
}