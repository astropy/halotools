
"""
Module expressing various default values of halotools. 

All hard-coding should be restricted to this module, whenever possible.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
import os, sys, configuration
import numpy as np

### Default halo catalog (used in read_nbody)
# The following parameters are used by the 
# simulation object in the read_nbody module
default_simulation_name = 'bolshoi'
default_halo_finder = 'rockstar'
default_scale_factor = 1.0003
default_numptcl = 2.0e5

# Default thresholds for mocks
default_luminosity_threshold = -20
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
min_permitted_conc = 1.0
max_permitted_conc = 25.0
default_dconc = 0.02
default_Npts_radius_array = 101
default_min_rad = 0.0001
default_max_rad = 1.0

### Default values specifying traditional quenching model
# Used by models in the halo_occupation module
default_quenching_parameters = {
    'quenching_abcissa' : [12,15],
    'central_quenching_ordinates' : [0.25,0.75], #polynomial coefficients determining quenched fraction of centrals
    'satellite_quenching_ordinates' : [0.25,0.75] #polynomial coefficients determining quenched fraction of centrals
    }

default_quiescence_dict = {
    'quiescence_abcissa' : [12,15], 
    'quiescence_ordinates' : [0.25, 0.75]
}

default_profile_dict = {
    'profile_abcissa' : [12,15], 
    'profile_ordinates' : [0.5,1]
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

# Set the default binsize used in assigning types to halos
default_halo_type_calculator_spacing=0.1

# Set the default secondary halo parameter used to generate assembly bias
default_assembias_key = 'vmax'

# URLs of websites hosting catalogs used by the package
aph_web_location = 'http://www.astro.yale.edu/aphearin/Data_files/'
behroozi_web_location = 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/'

# Convenience strings for the directory locations of the default catalogs (probably unnecessary)
halo_catalog_dirname = configuration.get_catalogs_dir('halos')
particle_catalog_dirname = configuration.get_catalogs_dir('particles')

# If the user requests a certain snapshot for halos or particles, 
# and the nearest available snapshot differs by more than the following amount, 
# the code will issue a warning.
scale_factor_difference_tol = 0.05

# At minimum, the following halo and galaxy properties 
# will be bound to each mock galaxy 
host_haloprop_prefix = 'halo_'
haloprop_list = ['id', 'pos', 'vel', 'mvir', 'rvir']
galprop_dict = {'gal_type':4,'pos':[4,4,4]}

testmode = False

haloprop_key_dict = {'prim_haloprop':'Mvir'}



