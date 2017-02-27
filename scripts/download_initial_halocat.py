#!/usr/bin/env python
"""
Command-line script to download the default halo catalog.

Executing this script with no arguments will download
a pre-processed hdf5 file storing a z=0 rockstar-based
subhalo catalog from the Bolshoi simulation. The catalog
will be stored in the following location on disk:
$HOME/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar

Executing this script also sets up your log of cached simulations.
The cache log is an ASCII file stored at the following location:
$HOME/.astropy/cache/halotools/halo_table_cache_log.txt

Manually deleting a line from this log erases the memory
of the corresponding catalog. In case the cache log becomes corrupted
for any reason, you can attempt to rebuild it
by running the following script:

$ python scripts/rebuild_halo_table_cache_log.py

"""

import os
from halotools.sim_manager import DownloadManager, sim_defaults
from halotools.custom_exceptions import HalotoolsError

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-overwrite",
    help="Overwrite the existing halo catalog (if present)",
    action="store_true")
parser.add_argument("-dirname",
    help="Absolute path to the dir to download the catalog. Default is std_cache_loc",
    default='std_cache_loc')
args = parser.parse_args()

existing_fname_error_msg = ("\n\nThe following filename already exists "
    "in your cache log: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "execute this script again but throw the ``-overwrite`` flag.\n\n")

simname = sim_defaults.default_simname
halo_finder = sim_defaults.default_halo_finder
redshift = sim_defaults.default_redshift
version_name = sim_defaults.default_version_name
ptcl_version_name = sim_defaults.default_ptcl_version_name

# Done parsing inputs

downman = DownloadManager()

##################################################################
# First check to see if the log has any matching entries before
# requesting the download
# This is technically redundant with the functionality in the downloading methods,
# but this makes it easier to issue the right error message
if args.overwrite is False:

    gen = downman.halo_table_cache.matching_log_entry_generator
    matching_halocats = list(
        gen(simname=simname, halo_finder=halo_finder,
            version_name=version_name, redshift=redshift, dz_tol=0.1))

    gen2 = downman.ptcl_table_cache.matching_log_entry_generator
    matching_ptcl_cats = list(
        gen2(simname=simname, version_name=ptcl_version_name,
            redshift=redshift, dz_tol=0.1))

    if len(matching_halocats) > 0:
        matching_fname = matching_halocats[0].fname
        raise HalotoolsError(existing_fname_error_msg % matching_fname)

    if len(matching_ptcl_cats) > 0:
        matching_fname = matching_ptcl_cats[0].fname
        raise HalotoolsError(existing_fname_error_msg % matching_fname)

##################################################################
# Call the download methods

new_halo_log_entry = downman.download_processed_halo_table(simname=simname,
    halo_finder=halo_finder, redshift=redshift, download_dirname=args.dirname,
        initial_download_script_msg=existing_fname_error_msg,
            overwrite=args.overwrite)

new_ptcl_log_entry = downman.download_ptcl_table(simname=simname,
    redshift=redshift, dz_tol=0.05, overwrite=args.overwrite, download_dirname=args.dirname,
    initial_download_script_msg=existing_fname_error_msg)

##################################################################


##################################################################
# Issue the success message
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
    "\nthe halo catalog is now stored in the following location:\n\n" +
    new_halo_log_entry.fname + "\n\n"
    "The "+ptcl_table_cache_basename+" cache log also has a single entry \n"
    "corresponding to a random downsampling of ~1e6 dark matter particles from the same snapshot; "
    "\nthe particle catalog is now stored in the following location:\n\n" +
    new_ptcl_log_entry.fname + "\n\n"
    "Both hdf5 files store an Astropy Table data structure. \n"
    "\nThe Halotools cache system allows you to \n"
    "load these catalogs into memory with the following syntax:\n\n"
    ">>> from halotools.sim_manager import CachedHaloCatalog\n"
    ">>> bolshoi_z0 = CachedHaloCatalog()\n"
    ">>> halos = bolshoi_z0.halo_table\n"
    ">>> particles = bolshoi_z0.ptcl_table\n\n")


print(msg)
print("\a\a")
