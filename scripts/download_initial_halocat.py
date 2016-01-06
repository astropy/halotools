#!/usr/bin/env python

"""Command-line script to download the default halo catalog"""

import os
from halotools.sim_manager import DownloadManager, sim_defaults

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", 
    help="Overwrite the existing halo catalog (if present)", 
    action="store_true")
args = parser.parse_args()

existing_fname_error_msg = ("\n\nThe following filename already exists in your cache directory: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "simply execute this script again but throwing ``--overwrite`` as a command-line flag.\n\n")

simname = sim_defaults.default_simname
halo_finder = sim_defaults.default_halo_finder
redshift = sim_defaults.default_redshift

downman = DownloadManager()

new_halo_log_entry = downman.download_processed_halo_table(simname = simname, 
    halo_finder = halo_finder, redshift = redshift, 
    initial_download_script_msg = existing_fname_error_msg, 
    overwrite = args.overwrite)
new_ptcl_log_entry = downman.download_ptcl_table(simname = simname, 
    redshift = redshift, dz_tol = 0.05, overwrite=args.overwrite, 
    initial_download_script_msg = existing_fname_error_msg)


cache_dirname = str(os.path.dirname(downman.halo_table_cache.cache_log_fname)).strip()
halo_table_cache_basename = str(os.path.basename(downman.halo_table_cache.cache_log_fname))
ptcl_table_cache_basename = str(os.path.basename(downman.ptcl_table_cache.cache_log_fname))

msg = (
    "By running the initial download script, you have set up the Halotools cache \n"
    "in the following location on disk:\n\n" + cache_dirname + "\n\n"
    "The directory now contains the following two cache log files: \n\n" + 
    str(downman.halo_table_cache.cache_log_fname) + "\n" + 
    str(downman.ptcl_table_cache.cache_log_fname) + "\n\n"
    "These two ASCII files maintain a record of the \nhalo and particle catalogs "
    "you use with Halotools.\n"
    "The "+halo_table_cache_basename+" cache log now has a single entry \n"
    "reflecting the default halo catalog you just downloaded; "
    "\nthe halo catalog is now stored in the following location:\n\n"
    + new_halo_log_entry.fname + "\n\n"
    "The "+ptcl_table_cache_basename+" cache log also has a single entry \n"
    "corresponding to a random downsampling of ~1e6 dark matter particles from the same snapshot; "
    "\nthe particle catalog is now stored in the following location:\n\n"
    + new_ptcl_log_entry.fname + "\n\n"
    "Both hdf5 files store an Astropy Table data structure. \n"
    "\nThe Halotools cache system allows you to \n"
    "load these catalogs into memory with the following syntax:\n\n"
    ">>> from halotools.sim_manager import OverhauledHaloCatalog\n"
    ">>> bolshoi_z0 = OverhauledHaloCatalog()\n"
    ">>> halos = bolshoi_z0.halo_table\n"
    ">>> particles = bolshoi_z0.ptcl_table\n\n")


print(msg)
print("\a\a")



