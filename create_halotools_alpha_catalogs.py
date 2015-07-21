#!/usr/bin/env python
import numpy as np

from halotools.sim_manager.catalog_manager import CatalogManager
from halotools.sim_manager.read_nbody_ascii import BehrooziASCIIReader


catman = CatalogManager()


flist = catman.raw_halo_tables_in_cache(simname='bolshoi', halo_finder='rockstar')
fname = flist[0]


column_bounds = [('halo_mpeak', 5e9, float("inf"))]

#reader = BehrooziASCIIReader(input_fname = fname, recompress=False, column_bounds=column_bounds)
reader = BehrooziASCIIReader(input_fname = fname, recompress=False, cuts_funcobj='nocut')
#reader = BehrooziASCIIReader(input_fname = fname, recompress=False)
#reader = BehrooziASCIIReader(input_fname = fname, recompress=False)

t = reader.read_halocat()

keys_to_keep = (['halo_scale_factor', 
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

for key in t.keys():
	if key not in keys_to_keep:
		del t[key]
t['halo_nfw_conc'] = t['halo_rvir'] / t['halo_rs']
del t['halo_rs']
t['halo_rvir'] /= 1000.

print("Number of halos = %i" % len(t))

t['halo_hostid'] = t['halo_upid']
host_mask = t['halo_upid'] == -1
sub_mask = np.invert(host_mask)
hosts = t[host_mask]
subs = t[sub_mask]

t['halo_hostid'][host_mask] = t['halo_id'][host_mask]

print("Number of subhalos = %i\n" % len(subs))



#external_cache_loc = os.path.abspath('/Volumes/NbodyDisk1/July19_new_catalogs')
