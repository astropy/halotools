"""
Module expressing various default values of the mock-making code.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)



### Default halo catalog (used in read_nbody)
# The following parameters are used by the 
# simulation object in the read_nbody module
default_simulation_name = 'bolshoi'
default_halo_finder = 'rockstar_V1.5'
default_scale_factor = 1.0003

# The following parameters are used by the 
# load_bolshoi_host_halos_fits method in 
# the read_nbody module. Still trying to 
# move away from this routine.
default_halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/presorted_host_halo_catalog.fits'
default_simulation_dict = {
	'catalog_filename':default_halo_catalog_filename,
	'Lbox':250.0,
	'scale_factor':1.0003,
	'particle_mass':1.35e8,
	'softening':1.0
}


default_luminosity_threshold = -20

# Small numerical value passed to the scipy Poisson number generator. 
# Used when executing a Monte Carlo realization of a Poission distribution 
# whose mean is formally zero, which causes the built-in 
# scipy method to raise an exception.
default_tiny_poisson_fluctuation = 1.e-20

# The numpy.digitize command has an annoying convention 
# such that if the value of the array being digitized, x, 
# is exactly equal to the bin boundary of the uppermost bin, 
# then numpy.digitize returns an index greater than the number 
# of bins. So by always setting the uppermost bin boundary to be 
# slightly larger than the largest value of x, this never happens.
default_bin_max_epsilon = 1.e-5

# Number of bins to use in the digitization of the NFW radial profile. 
# Used by HOD_Mock object in make_mocks module.
default_Npts_concentration_array = 1000
default_Npts_radius_array = 101


### Default values specifying traditional quenching model
# Used by models in the halo_occupation module
default_quenching_parameters = {
    'quenching_abcissa' : [12,13.5,15],
    'central_quenching_ordinates' : [0.35,0.7,0.95], #polynomial coefficients determining quenched fraction of centrals
    'satellite_quenching_ordinates' : [0.5,0.75,0.85] #polynomial coefficients determining quenched fraction of centrals
    }

default_assembias_parameters = {
	'assembias_abcissa' : [12,15],
	'satellite_assembias_ordinates' : [2,1],
	'central_assembias_ordinates' : [2,1]
	}

default_satcen_parameters = {
	'assembias_abcissa' : [12,13.5,15],
	'satellite_assembias_ordinates' : [1.5,1.25,0.5],
	'central_assembias_ordinates' : [1.0,1.0,1.0]
	}

default_halo_type_split = {
	'halo_type_split_abcissa' : [13,],
	'halo_type_split_ordinates' : [0.5,]
	}

default_halo_type_calculator_spacing=0.1


default_assembias_key = 'SCALE_50VMAX_MPEAK'






