#!/usr/bin/env python
import numpy as np
import os, sys

from halotools.sim_manager.catalog_manager import CatalogManager
from halotools.sim_manager.read_nbody_ascii import BehrooziASCIIReader
from halotools.sim_manager import sim_defaults
from halotools.utils import halocat_utils


#	external_cache_loc = os.path.abspath('/Volumes/NbodyDisk1/July19_new_catalogs')

def main(input_cache_loc):

	catman = CatalogManager()

	def process_and_store_result(fname):

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


	raw_halo_files_to_process = catman.raw_halo_tables_in_cache(external_cache_loc = input_cache_loc)
	for fname in raw_halo_files_to_process:
		process_and_store_result(fname)



###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv[1])




