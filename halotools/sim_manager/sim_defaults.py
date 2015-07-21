"""
Module expressing various default settings of 
the simulation manager sub-package. 

All hard-coding should be restricted to this module, whenever possible.
"""

import os, sys
import numpy as np

from astropy import cosmology
from . import raw_halocat_column_info
from ..halotools_exceptions import UnsupportedSimError, HalotoolsCacheError

### Default halo catalog (used in read_nbody)
# The following parameters are used by the 
# simulation object in the read_nbody module
default_simname = 'bolshoi'
default_halo_finder = 'rockstar'
default_numptcl = 2.0e5
default_redshift = 0.0

Num_ptcl_requirement = 300
mass_like_variable_to_apply_cut = 'halo_mpeak'

default_cosmology = cosmology.WMAP5

# URLs of websites hosting catalogs used by the package
processed_halo_tables_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/halo_catalogs'
ptcl_tables_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/particle_catalogs'
default_version_name = 'halotools.alpha.version0'

default_cache_location = 'pkg_default'


############################################################
### Current versions of dtype and header 
# for reading ASCII data are defined below 
# Definitions are in terms of historical information 
#defined in raw_halocat_column_info 
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
dtype_consuelo_rockstar = raw_halocat_column_info.dtype_slac_consuelo_rockstar_july19_2015
header_consuelo_rockstar = raw_halocat_column_info.header_slac_consuelo_rockstar_july19_2015
#
dtype_bolshoi_bdm = raw_halocat_column_info.dtype_slac_bolshoi_bdm_july19_2015
header_bolshoi_bdm = raw_halocat_column_info.header_slac_bolshoi_bdm_july19_2015

def return_dtype_and_header(simname, halo_finder):
	if halo_finder == 'rockstar':
		if simname == 'bolshoi':
			return dtype_bolshoi_rockstar, header_bolshoi_rockstar
		elif simname == 'bolplanck':
			return dtype_bolplanck_rockstar, header_bolplanck_rockstar
		elif simname == 'multidark':
			return dtype_multidark_rockstar, header_multidark_rockstar
		elif simname == 'consuelo':
			return dtype_consuelo_rockstar, header_consuelo_rockstar
		else:
			raise UnsupportedSimError(simname)
	elif halo_finder == 'bdm':
		if simname == 'bolshoi':
			return dtype_bolshoi_bdm, header_bolshoi_bdm
	else:
		msg = "Unsupported combination of halo_finder = ``%s`` and simname = ``%s``"
		raise HalotoolsCacheError(msg % (halo_finder, simname))










############################################################



default_ascii_columns_to_keep = (['halo_scale_factor', 
	'halo_id',
	'halo_pid',
	'halo_upid',
	'halo_phantom',
	'halo_mvir',
	'halo_rvir',
	'halo_rs',
	'halo_vrms',
	'halo_scale_factor_lastmm',
	'halo_vmax',
	'halo_x',
	'halo_y',
	'halo_z',
	'halo_vx',
	'halo_vy',
	'halo_vz',
	'halo_jx',
	'halo_jy',
	'halo_jz',
	'halo_spin',
	'halo_id_tree_root',
	'halo_rs_klypin',
	'halo_xoff',
	'halo_voff',
	'halo_spin_bullock',
	'halo_b_to_a',
	'halo_c_to_a',
	'halo_axisA_x',
	'halo_axisA_y',
	'halo_axisA_z',
	'halo_t_by_u',
	'halo_macc',
	'halo_mpeak',
	'halo_vacc',
	'halo_vpeak',
	'halo_halfmass_scale',
	'halo_dmvir_dt_inst',
	'halo_dmvir_dt_100myr',
	'halo_dmvir_dt_tdyn',
	'halo_scale_factor_mpeak',
	'halo_scale_factor_lastacc',
	'halo_scale_factor_firstacc',
	'halo_mvir_firstacc',
	'halo_vmax_firstacc',
	'halo_vmax_mpeak'
	])





