"""
Module expressing various default settings of 
the empirical models sub-package. 

All hard-coding should be restricted to this module, whenever possible.
"""

import os, sys
import numpy as np

from astropy import cosmology

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

# Number of bins to use in the lookup table attached to the NFWProfile. 
# Used primarily by HODMockFactory.
min_permitted_conc = 0.1
max_permitted_conc = 30.0
default_dconc = 0.025

default_Npts_radius_array = 101
default_lograd_min = -4
default_lograd_max = 0
profile_table_radius_array_dict = {
    'logrmin' : default_lograd_min, 
    'logrmax' : default_lograd_max, 
    'npts' : default_Npts_radius_array
}
conc_mass_relation_key = 'dutton_maccio14'

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




# At minimum, the following halo and galaxy properties 
# will be bound to each mock galaxy 
host_haloprop_prefix = 'halo_'
galprop_prefix = 'gal_'
haloprop_list = ['haloid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mvir', 'rvir']

# obsolete now
#galprop_dict = {'gal_type':4,'pos':[4,4,4]}

prim_haloprop_key = 'mvir'
sec_haloprop_key = 'vmax'
halo_boundary = 'rvir'
haloprop_key_dict = {'prim_haloprop_key':prim_haloprop_key, 'halo_boundary':halo_boundary}

assembias_haloprop_key_dict = {
    'prim_haloprop_key':'mvir', 
    'halo_boundary':'rvir',
    'sec_haloprop_key':'vmax'
    }









