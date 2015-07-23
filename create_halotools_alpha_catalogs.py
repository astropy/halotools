#!/usr/bin/env python

""" This script is responsible for generating all the 
processed halo catalogs made publicly available with Halotools. 
"""

import numpy as np
import os, sys

from halotools.sim_manager.catalog_manager import CatalogManager
from halotools.sim_manager.read_nbody_ascii import BehrooziASCIIReader
from halotools.sim_manager import sim_defaults
from halotools.utils import halocat_utils

catman = CatalogManager()

### External disk location to which hipacc files were downloaded on July 19, 2015
input_cache_loc = os.path.abspath('/Volumes/NbodyDisk1/July19_new_catalogs')


def process_and_store_result(fname):
	""" Function reads ascii data from the input fname, 
	makes the default cut defined in the `halotools.sim_manager.BehrooziASCIIReader` class,
	adds a few convenience columns, and writes the result to the cache directory the local 
	cache directory on Muffuletta. These processed files were then uploaded to the following url:
	http://www.astro.yale.edu/aphearin/Data_files/halo_catalogs. 
	"""

	reader = BehrooziASCIIReader(input_fname = fname, overwrite=True)
	halo_table = reader.read_halocat()

	keys_to_keep = sim_defaults.default_ascii_columns_to_keep
	for key in halo_table.keys():
		if key not in keys_to_keep:
			del halo_table[key]
	halo_table['halo_nfw_conc'] = halo_table['halo_rvir'] / halo_table['halo_rs']
	del halo_table['halo_rs']
	halo_table['halo_rvir'] /= 1000. # convert rvir to Mpc

	halo_table['halo_hostid'] = halo_table['halo_upid']
	host_mask = halo_table['halo_upid'] == -1
	halo_table['halo_hostid'][host_mask] = halo_table['halo_id'][host_mask]

	halo_table['host_halo_status'] = halocat_utils.host_status(halo_table)

	catman.store_newly_processed_halo_table(
		halo_table, reader, sim_defaults.default_version_name, overwrite=True)


### Process the Bolshoi-Planck Rockstar catalogs (z = 0, 0.5, 1, 2)
raw_halo_files_to_process = catman.raw_halo_tables_in_cache(
	external_cache_loc = input_cache_loc, simname = 'bolplanck')
for fname in raw_halo_files_to_process:
	print("\n\n\n...Calling the process_and_store_result function for the following filename: \n%s" % fname)
	process_and_store_result(fname)


### Process the Bolshoi catalogs, both Rockstar and BDM (z = 0, 0.5, 1, 2)
raw_halo_files_to_process = catman.raw_halo_tables_in_cache(
	external_cache_loc = input_cache_loc, simname = 'bolshoi')
for fname in raw_halo_files_to_process:
	print("\n\n\n...Calling the process_and_store_result function for the following filename: \n%s" % fname)
	process_and_store_result(fname)


### Process the MultiDark Rockstar catalogs (z = 0, 0.5, 1, 2)
raw_halo_files_to_process = catman.raw_halo_tables_in_cache(
	external_cache_loc = input_cache_loc, simname = 'multidark')
for fname in raw_halo_files_to_process:
	print("\n\n\n...Calling the process_and_store_result function for the following filename: \n%s" % fname)
	process_and_store_result(fname)

### Process the Consuelo Rockstar catalogs (z = 0, 0.5, 1, 2)
raw_halo_files_to_process = catman.raw_halo_tables_in_cache(
	external_cache_loc = input_cache_loc, simname = 'consuelo')
for fname in raw_halo_files_to_process:
	print("\n\n\n...Calling the process_and_store_result function for the following filename: \n%s" % fname)
	process_and_store_result(fname)














