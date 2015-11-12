"""
Module expressing various default settings of the empirical modeling sub-package. 

All values hard-coded here appear as unique variables throughout the entire Halotools code base. 
This allows you to customize your default settings and be guaranteed that whatever changes you make 
will correctly propagate to all relevant behavior. See the in-line comments in the 
``halotools/empirical_models/model_defaults.py`` source code for 
descriptions of the purpose of each variable defined in this module. 
"""

__all__ = ['get_halo_boundary_key', 'get_halo_mass_key']

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


### Default values specifying traditional quenching model
# Used by models in the halo_occupation module

default_profile_dict = {
    'profile_abcissa' : [12,15], 
    'profile_ordinates' : [0.5,1]
}

# Set the default binsize used in assigning types to halos
default_halo_type_calculator_spacing=0.1

# Set the default secondary halo parameter used to generate assembly bias
default_assembias_key = 'halo_vmax'

default_smhm_scatter = 0.2
default_smhm_haloprop = 'halo_mpeak'
default_binary_galprop_haloprop = default_smhm_haloprop

# At minimum, the following halo and galaxy properties 
# will be bound to each mock galaxy 
host_haloprop_prefix = 'halo_'
galprop_prefix = 'gal_'
default_haloprop_list_inherited_by_mock = (
    ['halo_id', 'halo_x', 'halo_y', 'halo_z', 
    'halo_vx', 'halo_vy', 'halo_vz', 
    'halo_mvir', 'halo_rvir']
    )

prim_haloprop_key = 'halo_mvir'
sec_haloprop_key = 'halo_nfw_conc'

halo_mass_definition = 'vir'
def get_halo_boundary_key(mdef):
    """ For the input mass definition, 
    return the string used to access halo table column 
    storing the halo radius. 

    Parameters 
    -----------
    mdef: str
        String specifying the halo mass definition, e.g., 'vir' or '200m'. 

    Returns 
    --------
    radius_key : str
    """
    return 'halo_r'+mdef
def get_halo_mass_key(mdef):
    """ For the input mass definition, 
    return the string used to access halo table column 
    storing the halo mass. 

    Parameters 
    -----------
    mdef: str
        String specifying the halo mass definition, e.g., 'vir' or '200m'. 

    Returns 
    --------
    mass_key : str
    """
    return 'halo_m'+mdef

# Number of bins to use in the lookup table attached to the NFWProfile. 
# Used primarily by HODMockFactory.
min_permitted_conc = 1
max_permitted_conc = 30.0
default_high_prec_dconc = 0.025

Npts_radius_table = 101
default_lograd_min = -3
default_lograd_max = 0
conc_mass_model = 'direct_from_halo_catalog'
concentration_key = 'halo_nfw_conc'


default_rbins = np.logspace(-1, 1.25, 15)
default_nptcls = 1e5

default_b_perp = 0.2
default_b_para = 0.75






