#!/usr/bin/env python
"""Command-line script to download any
Halotools-provided halo catalog.

This script should be called with four positional arguments:

    1. simname
    2. halo_finder
    3. version_name
    4. redshift

To see what options are available for download,
run this script with no arguments but throw the help flag:

$ python scripts/download_additional_halocat.py -h

This script will download your halo catalogs to the
following location on disk:

$HOME/.astropy/cache/halotools/halo_catalogs/simname/halo_finder

With each download, your cache log is updated so that Halotools creates
a persistent memory of where your simulations are located.
Your cache log is an ASCII file located here:

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
    help="Overwrite the existing halo catalog (if present). ",
    action="store_true")

parser.add_argument("-dirname",
    help="Absolute path to the dir to download the catalog. Default is std_cache_loc",
        default='std_cache_loc')

parser.add_argument("-ptcls_only",
    help=("Only download a random downsampling of 1e6 dark matter particles from the snapshot."
        "Downsampled particles are necessary to calculate galaxy-galaxy lensing. "
        "z>0 particles are currently not available for bolshoi or multidark."),
    action="store_true")

parser.add_argument("-halos_only",
    help=("Only download the halo catalog data of the snapshot, "
        "and ignore the downsampled particle data. "
        "Since z>0 particles are currently not available "
        "for bolshoi or multidark, "
        "this flag must be thrown in order to download z>0 halo catalogs "
        "for these simulations."),
    action="store_true")

parser.add_argument("simname", type=str,
    choices=['bolshoi', 'bolplanck', 'multidark', 'consuelo'],
    help="Nickname of the simulation. ")

parser.add_argument("halo_finder", type=str, help="Nickname of the halo-finder. "
    "The `bdm` option is only available for `bolshoi`. ",
    choices=['rockstar', 'bdm'])

parser.add_argument("version_name", type=str,
    choices=['halotools_v0p4', 'most_recent'],
    help="Processing version of the requested catalog. "
    "Selecting `most_recent` will automatically choose the most up-to-date catalogs. ")

parser.add_argument("redshift", type=float, help="Redshift of the snapshot. "
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
if args.version_name == 'most_recent':
    version_name = sim_defaults.default_version_name

ptcl_version_name = sim_defaults.default_ptcl_version_name

if args.ptcls_only is True:
    download_halos = False
else:
    download_halos = True

if args.halos_only is True:
    download_ptcls = False
else:
    download_ptcls = True

#  Raise a special message if high-redshift bolshoi or multidark particles are requested.
hiz_no_ptcls_error_msg = (
    "\nHigh-redshift particles are not available for the " + simname + " simulation.\n"
    "To download the high-z " + simname + " halos you have requested, "
    "throw the -halos_only flag.\n")

if download_ptcls is True:
    if (redshift > 0.1) & ((simname == 'bolshoi') or (simname == 'multidark')):
        raise ValueError(hiz_no_ptcls_error_msg)

# Done parsing inputs

downman = DownloadManager()

##################################################################
# First check to see if the log has any matching entries before
# requesting the download
# This is technically redundant with the functionality in the downloading methods,
# but this makes it easier to issue the right error message
if args.overwrite is False:

    if download_halos is True:

        gen = downman.halo_table_cache.matching_log_entry_generator
        matching_halocats = list(
            gen(simname=simname, halo_finder=halo_finder,
                version_name=version_name, redshift=redshift, dz_tol=0.1))

        if len(matching_halocats) > 0:
            matching_fname = matching_halocats[0].fname
            raise HalotoolsError(existing_fname_error_msg % matching_fname)

    if download_ptcls is True:

        gen2 = downman.ptcl_table_cache.matching_log_entry_generator
        matching_ptcl_cats = list(
            gen2(simname=simname, version_name=ptcl_version_name,
                redshift=redshift, dz_tol=0.1))

        if len(matching_ptcl_cats) > 0:
            matching_fname = matching_ptcl_cats[0].fname
            raise HalotoolsError(existing_fname_error_msg % matching_fname)

##################################################################

##################################################################
# Call the download methods
if download_ptcls is True:
    new_ptcl_log_entry = downman.download_ptcl_table(simname=simname,
        redshift=redshift, dz_tol=0.05, overwrite=args.overwrite, download_dirname=args.dirname,
        initial_download_script_msg=existing_fname_error_msg)

if download_halos is True:
    new_halo_log_entry = downman.download_processed_halo_table(simname=simname,
        halo_finder=halo_finder, redshift=redshift, download_dirname=args.dirname,
        initial_download_script_msg=existing_fname_error_msg,
        overwrite=args.overwrite)

##################################################################


##################################################################
# Issue the success message

cache_dirname = str(os.path.dirname(downman.halo_table_cache.cache_log_fname)).strip()
halo_table_cache_basename = str(os.path.basename(downman.halo_table_cache.cache_log_fname))
ptcl_table_cache_basename = str(os.path.basename(downman.ptcl_table_cache.cache_log_fname))

msg = (
    "The Halotools cache is stored in the following location on disk:\n\n" + cache_dirname + "\n\n"
    "That directory contains the following two cache log files: \n\n" +
    str(downman.halo_table_cache.cache_log_fname) + "\n" +
    str(downman.ptcl_table_cache.cache_log_fname) + "\n\n"
    "These two ASCII files maintain a record of the \nhalo and particle catalogs "
    "you use with Halotools.\n")

if download_halos is True:
    msg += ("The " + halo_table_cache_basename + " cache log now has an entry \n"
        "corresponding to the newly downloaded halo catalog,\n"
        "which is stored in the following location:\n\n" +
        new_halo_log_entry.fname + "\n\n")

if download_ptcls is True:
    msg += ("The " + ptcl_table_cache_basename + " cache log now has an entry \n"
        "corresponding to a random downsampling of \n"
        "~1e6 dark matter particles from the requested snapshot; "
        "\nthe particle catalog is stored in the following location:\n\n" +
        new_ptcl_log_entry.fname + "\n\n")

msg += ("This data is in the form of an hdf5 files store an Astropy Table data structure. \n"
    "\nThe Halotools cache system allows you to \n"
    "load these catalogs into memory with the following syntax:\n\n"
    ">>> from halotools.sim_manager import CachedHaloCatalog\n"
    ">>> halocat = CachedHaloCatalog(simname='" + simname + "', "
    "halo_finder='" + halo_finder + "', redshift=" + str(redshift) +
    ", version_name='" + version_name + "')\n")

if download_halos is True:
    msg += ">>> halos = halocat.halo_table\n"

if download_ptcls is True:
    msg += ">>> particles = halocat.ptcl_table\n\n"


print(msg)
print("\a\a")
