#!/usr/bin/env python
import numpy as np

from halotools.sim_manager.catalog_manager import CatalogManager
from halotools.sim_manager.read_nbody_ascii import BehrooziASCIIReader


catman = CatalogManager()


flist = catman.raw_halo_tables_in_cache(simname='bolshoi', halo_finder='bdm')
fname = flist[0]



#reader = BehrooziASCIIReader(input_fname = fname, recompress=False, cuts_funcobj='nocut')
#reader = BehrooziASCIIReader(input_fname = fname, recompress=False)
reader = BehrooziASCIIReader(input_fname = fname, recompress=False, column_bounds=column_bounds)

t = reader.read_halocat()

bolshoi_bdm_keys_to_keep = (['halo_scale_factor', 
	'halo_id',
	'halo_id_desc',
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
	'halo_xoff',
	'halo_2K/Ep-1',
	'halo_b_to_a',
	'halo_c_to_a',
	'halo_axisA_x',
	'halo_axisA_y',
	'halo_axisA_z',
	'halo_macc',
	'halo_mpeak',
	'halo_vacc',
	'halo_vpeak'
	])

for key in t.keys():
	if key not in bolshoi_bdm_keys_to_keep:
		del t[key]






#external_cache_loc = os.path.abspath('/Volumes/NbodyDisk1/July19_new_catalogs')
