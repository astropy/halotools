#!/usr/bin/env python

"""Command-line script to download the default halo catalog"""

import os
from halotools.sim_manager import DownloadManager, sim_defaults
from halotools.custom_exceptions import HalotoolsError

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-overwrite", 
    help="Overwrite the existing halo catalog (if present)", 
    action="store_true")

parser.add_argument("-ptcls_only", 
    help="Only download the particle data of the snapshot", 
    action="store_true")

parser.add_argument("-halos_only", 
    help="Only download the halo catalog data of the snapshot", 
    action="store_true")

parser.add_argument("simname", type = str, 
	choices = ['bolshoi', 'bolplanck', 'multidark', 'consuelo'], 
	help = "Nickname of the simulation")

parser.add_argument("halo_finder", type = str, help = "Nickname of the halo-finder. "
	"The `bdm` option is only available for `bolshoi`. ", 
	choices = ['rockstar', 'bdm'])

parser.add_argument("version_name", type = str, 
	choices = ['halotools_alpha_version1', 'most_recent'])

parser.add_argument("redshift", type = float, help = "Redshift of the snapshot. "
	"Options are 0, 0.5, 1 and 2, with slight variations from simulation to simulation.")

args = parser.parse_args()

existing_fname_error_msg = ("\n\nThe following filename already exists "
    "in your cache log: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "execute this script again but throw the ``-overwrite`` flag.\n\n")

simname = args.simname
halo_finder = args.halo_finder
version_name = args.version_name
redshift = args.redshift
if version == 'most_recent': version = sim_defaults.default_version_name


if args.ptcls_only is True: download_halos = False
if args.halos_only is True: download_ptcls = False


downman = DownloadManager()

##################################################################
# First check to see if the log has any matching entries before 
# requesting the download 
# This is technically redundant with the functionality in the downloading methods, 
# but this makes it easier to issue the right error message


raise HalotoolsError("LEFT OFF HERE")


if args.overwrite == False:
	pass


##################################################################

##################################################################
### Call the download method

new_halo_log_entry = downman.download_processed_halo_table(simname = simname, 
    halo_finder = halo_finder, redshift = redshift, 
    initial_download_script_msg = existing_fname_error_msg, 
    overwrite = args.overwrite)
new_ptcl_log_entry = downman.download_ptcl_table(simname = simname, 
    redshift = redshift, dz_tol = 0.05, overwrite=args.overwrite, 
    initial_download_script_msg = existing_fname_error_msg)


##################################################################




##################################################################
### Issue the success message
cache_dirname = str(os.path.dirname(downman.halo_table_cache.cache_log_fname)).strip()
halo_table_cache_basename = str(os.path.basename(downman.halo_table_cache.cache_log_fname))
ptcl_table_cache_basename = str(os.path.basename(downman.ptcl_table_cache.cache_log_fname))

msg = (
    "The Halotools cache is stored in the following location on disk:\n\n" + cache_dirname + "\n\n"
    "The directory contains the following two cache log files: \n\n" + 
    str(downman.halo_table_cache.cache_log_fname) + "\n" + 
    str(downman.ptcl_table_cache.cache_log_fname) + "\n\n"
    "These two ASCII files maintain a record of the \nhalo and particle catalogs "
    "you use with Halotools.\n"
    "The "+halo_table_cache_basename+" cache log now has an entry \n"
    "reflecting the halo catalog you just downloaded; "
    "\nthe halo catalog is now stored in the following location:\n\n"
    + new_halo_log_entry.fname + "\n\n"
    "The "+ptcl_table_cache_basename+" cache log also has an entry \n"
    "corresponding to a random downsampling of ~1e6 dark matter particles from the same snapshot; "
    "\nthe particle catalog is stored in the following location:\n\n"
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



