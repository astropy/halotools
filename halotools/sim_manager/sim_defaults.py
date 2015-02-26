"""
Module expressing various default settings of 
the simulation manager sub-package. 

All hard-coding should be restricted to this module, whenever possible.
"""

import os, sys
import numpy as np

from astropy import cosmology

import configuration

### Default halo catalog (used in read_nbody)
# The following parameters are used by the 
# simulation object in the read_nbody module
default_simulation_name = 'bolshoi'
default_halo_finder = 'rockstar'
default_scale_factor = 1.0003
default_numptcl = 2.0e5

# URLs of websites hosting catalogs used by the package
aph_web_location = 'http://www.astro.yale.edu/aphearin/Data_files/'
behroozi_web_location = 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/'

# Convenience strings for the directory locations of the default catalogs (probably unnecessary)
halo_catalog_dirname = configuration.get_catalogs_dir('halos')
particle_catalog_dirname = configuration.get_catalogs_dir('particles')

default_redshift = 0.0
default_cosmology = cosmology.WMAP5

# If the user requests a certain snapshot for halos or particles, 
# and the nearest available snapshot differs by more than the following amount, 
# the code will issue a warning.
scale_factor_difference_tol = 0.05








