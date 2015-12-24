#!/usr/bin/env python

import numpy as np 
from halotools.sim_manager import RockstarHlistReader, OverhauledHaloCatalog
from halotools.sim_manager import TabularAsciiReader
from halotools.sim_manager import manipulate_cache_log
from astropy.table import Table 
import os
from halotools.sim_manager.read_rockstar_hlists import _infer_redshift_from_input_fname

num_ptcl_cut = 300.
processing_notes = 'Catalog only contains (sub)halos with present-day virial mass greater than 300 particles'

bolshoi_columns_to_keep_dict = ({
    'halo_scale_factor': (0, 'f4'), 
    'halo_id': (1, 'i8'), 
    'halo_pid': (5, 'i8'), 
    'halo_upid': (6, 'i8'), 
    'halo_mvir': (10, 'f4'), 
    'halo_rvir': (11, 'f4'), 
    'halo_rs': (12, 'f4'), 
    'halo_scale_factor_last_mm': (15, 'f4'), 
    'halo_vmax': (16, 'f4'), 
    'halo_x': (17, 'f4'), 
    'halo_y': (18, 'f4'), 
    'halo_z': (19, 'f4'), 
    'halo_vx': (20, 'f4'), 
    'halo_vy': (21, 'f4'), 
    'halo_vz': (22, 'f4'), 
    'halo_jx': (23, 'f4'), 
    'halo_jy': (24, 'f4'), 
    'halo_jz': (25, 'f4'), 
    'halo_spin': (26, 'f4'), 
    'halo_m200b': (37, 'f4'), 
    'halo_m200c': (38, 'f4'), 
    'halo_m500c': (39, 'f4'), 
    'halo_m2500c': (40, 'f4'), 
    'halo_xoff': (41, 'f4'), 
    'halo_voff': (42, 'f4'), 
    'halo_b_to_a': (44, 'f4'), 
    'halo_c_to_a': (45, 'f4'), 
    'halo_axisA_x': (46, 'f4'), 
    'halo_axisA_y': (47, 'f4'), 
    'halo_axisA_z': (48, 'f4'), 
    'halo_t_by_u': (54, 'f4'), 
    'halo_m_pe_behroozi': (55, 'f4'), 
    'halo_m_pe_diemer': (56, 'f4'), 
    'halo_macc': (57, 'f4'), 
    'halo_mpeak': (58, 'f4'), 
    'halo_vacc': (59, 'f4'), 
    'halo_vpeak': (60, 'f4'), 
    'halo_halfmass_scale_factor': (61, 'f4'), 
    'halo_dmvir_dt_inst': (62, 'f4'), 
    'halo_dmvir_dt_100myr': (63, 'f4'), 
    'halo_dmvir_dt_tdyn': (64, 'f4'), 
    'halo_scale_factor_mpeak': (67, 'f4'), 
    'halo_scale_factor_lastacc': (68, 'f4'), 
    'halo_scale_factor_firstacc': (69, 'f4'), 
    'halo_mvir_firstacc': (70, 'f4'), 
    'halo_vmax_firstacc': (71, 'f4'), 
    'halo_vmax_mpeak': (72, 'f4')
    })

def process_catalog(simname, halo_finder, fname, particle_mass, Lbox, columns_to_keep_dict):

    redshift = _infer_redshift_from_input_fname(fname)
    row_cut_min_dict = {'halo_mvir': num_ptcl_cut*particle_mass}

    print("\n...working on redshift = " + str(redshift) + "...\n")
    reader = RockstarHlistReader(input_fname = fname, 
        columns_to_keep_dict=columns_to_keep_dict, 
        output_fname='std_cache_loc', 
        simname=simname, 
        halo_finder=halo_finder, 
        redshift=redshift, 
        version_name='halotools_alpha_version1', 
        Lbox=Lbox, 
        particle_mass=particle_mass, 
        row_cut_min_dict = row_cut_min_dict, 
        processing_notes = processing_notes, 
        overwrite=True)

    reader.read_halocat(write_to_disk = True, update_cache_log = True)
    del reader



#### BOLSHOI #####
bolshoi_Lbox = 250.
bolshoi_particle_mass = 1.35e8
dirname = '/Volumes/NbodyDisk1/July19_new_catalogs/raw_halo_catalogs/bolshoi/rockstar'
basename_list = ('hlist_0.54435.list', 'hlist_0.67035.list', 'hlist_1.00035.list')
fname_generator = (os.path.join(dirname, basename) for basename in basename_list)

for fname in fname_generator:
    process_catalog('bolshoi', 'rockstar', fname, 
        bolshoi_particle_mass, bolshoi_Lbox, bolshoi_columns_to_keep_dict)






