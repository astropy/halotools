
"""
Module expressing various default values of the mock-making code.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
import os
import sys

### Default halo catalog (used in read_nbody)
# The following parameters are used by the 
# simulation object in the read_nbody module
default_simulation_name = 'bolshoi'
default_halo_finder = 'rockstar_V1.5'
default_scale_factor = 1.0003
default_redshift = 0.0

### Default particle data (used in read_nbody)
# The following parameters are used by the 
# particles object in the read_nbody module
default_size_particle_data = '2e5'


default_luminosity_threshold = -20

# Default stellar mass threshold for stellar mass
# limited samples
default_stellar_mass_threshold = 10.5

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
    'quenching_abcissa' : [12,15],
    'central_quenching_ordinates' : [0.5,0.5], #polynomial coefficients determining quenched fraction of centrals
    'satellite_quenching_ordinates' : [0.5,0.5] #polynomial coefficients determining quenched fraction of centrals
    }

default_occupation_assembias_parameters = {
	'assembias_abcissa' : [12,15],
	'satellite_assembias_ordinates' : [0.5,0.5],
	'central_assembias_ordinates' : [2,2]
	}

default_quenching_assembias_parameters = {
	'quenching_assembias_abcissa' : [12,13.5,15],
	'satellite_quenching_assembias_ordinates' : [0.05,0.05,100],
	'central_quenching_assembias_ordinates' : [20,-20,20]
	}

default_satcen_parameters = {
	'assembias_abcissa' : [12,13.5,15],
	'satellite_assembias_ordinates' : [1.5,1.25,0.5],
	'central_assembias_ordinates' : [1.0,1.0,1.0]
	}

default_halo_type_split = {
	'halo_type_split_abcissa' : [12,13,14,15],
	'halo_type_split_ordinates' : [0.1,0.9,0.5,0.9]
	}

default_halo_type_calculator_spacing=0.1

default_assembias_key = 'VMAX'

relative_filepath = '/CATALOGS/'
catalog_dirname = os.path.join(os.path.dirname(__file__),relative_filepath) 



# Filenames of pointing to various simulation data 
bolshoi_z0_2e5_particles_filename='http://www.astro.yale.edu/aphearin/Data_files/bolshoi_2e5_particles_a1.0003.fits'
bolshoi_z0_halos_filename='http://www.astro.yale.edu/aphearin/Data_files/bolshoi_a1.0003_rockstar_V1.5_host_halos.fits'


# Set the defaults to bolshoi at z=0
default_halo_catalog_filename=bolshoi_z0_halos_filename
default_particle_catalog_filename=bolshoi_z0_2e5_particles_filename


aph_web_location = 'http://www.astro.yale.edu/aphearin/Data_files/'
behroozi_web_location = 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/'





scale_factor_difference_tol = 0.05






