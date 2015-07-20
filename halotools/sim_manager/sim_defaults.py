"""
Module expressing various default settings of 
the simulation manager sub-package. 

All hard-coding should be restricted to this module, whenever possible.
"""

import os, sys
import numpy as np

from astropy import cosmology
from . import raw_halocat_column_info

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
processed_halo_tables_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/halo_catalogs'
ptcl_tables_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/particle_catalogs'
default_version_name = 'halotools.official.version'

############################################################
### Current versions of dtype and header 
# for reading ASCII data are defined below 
# Definitions are in terms of historical information 
#defined in raw_halocat_column_info 
dtype_consuelo_rockstar = raw_halocat_column_info.dtype_slac_consuelo_rockstar_july19_2015
header_consuelo_rockstar = raw_halocat_column_info.header_slac_consuelo_rockstar_july19_2015
#
dtype_bolshoi_rockstar = raw_halocat_column_info.dtype_slac_bolshoi_rockstar_july19_2015
header_bolshoi_rockstar = raw_halocat_column_info.header_slac_bolshoi_rockstar_july19_2015
#
dtype_bolplanck_rockstar = raw_halocat_column_info.dtype_slac_bolplanck_rockstar_july19_2015
header_bolplanck_rockstar = raw_halocat_column_info.header_slac_bolplanck_rockstar_july19_2015
#
dtype_multidark_rockstar = raw_halocat_column_info.dtype_slac_multidark_rockstar_july19_2015
header_multidark_rockstar = raw_halocat_column_info.header_slac_multidark_rockstar_july19_2015
#
dtype_bolshoi_bdm = raw_halocat_column_info.dtype_slac_bolshoi_bdm_july19_2015
header_bolshoi_bdm = raw_halocat_column_info.header_slac_bolshoi_bdm_july19_2015
############################################################

