"""
Module expressing various default settings of 
the simulation manager sub-package. 

All hard-coding should be restricted to this module, whenever possible.
"""

import os, sys
import numpy as np

from astropy import cosmology

raw_halocat_cache_dir = 'pkg_default'
processed_halocat_cache_dir = 'pkg_default'
particles_cache_dir = 'pkg_default'

### Default halo catalog (used in read_nbody)
# The following parameters are used by the 
# simulation object in the read_nbody module
default_simname = 'bolshoi'
default_halo_finder = 'rockstar'
default_numptcl = 2.0e5
default_redshift = 0.0
Num_ptcl_requirement = 300

default_cosmology = cosmology.WMAP5

# URLs of websites hosting catalogs used by the package
processed_halocats_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/halo_catalogs'
ptcl_cats_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/particle_catalogs'
default_version_name = 'halotools.official.version'
