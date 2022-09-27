#!/usr/bin/env python
"""
Command-line script to rebuild the halo table cache log
in the event that the log is lost or becomes corrupted.

The cache log is an ASCII file stored at the following location:
$HOME/.astropy/cache/halotools/halo_table_cache_log.txt

This file is used by Halotools to create a persistent memory of
the locations of your halo catalogs on disk. If you accidentally
delete this file or edit it in a way that makes it unreadable
by Halotools, you can run this script to attempt to rebuild the log.

The first thing this script does is attempt to extract all directories
that may be listed in the existing log. This list of directories is
supplemented with the default location where halo catalogs are stored,
$HOME/.astropy/cache/halotools/halo_catalogs.
Then a list of absolute paths to potential halo catalogs
is built by searching for any hdf5 file appearing in any of
these directories or their subdirectories. Each such hdf5 file is
tested for compatibility with the Halotools catalog format.
Files passing all tests are added to a newly created log;
the names of files failing any test are added to the following ASCII file:

$HOME/.astropy/cache/halotools/rejected_halo_table_filenames.txt

The previously existing log is saved to the following location
so that no information is destroyed in the rebuilding process:

$HOME/.astropy/cache/halotools/corrupted_halo_table_cache_log.txt
"""

import argparse, os, fnmatch
from astropy.table import Table
import numpy as np
from time import time

try:
    import h5py
except ImportError:
    msg = ("\nMust have h5py installed to use the rebuild_halo_table_cache_log script.\n")
    raise ImportError(msg)

from halotools.custom_exceptions import HalotoolsError
from halotools.sim_manager import HaloTableCache
from halotools.sim_manager.halo_table_cache_log_entry import HaloTableCacheLogEntry

old_cache = HaloTableCache()
old_cache_log_exists = os.path.isfile(old_cache.cache_log_fname)

cache_log_dirname = os.path.dirname(old_cache.cache_log_fname)
corrupted_cache_log_basename = 'corrupted_halo_table_cache_log.txt'
corrupted_cache_log_fname = os.path.join(cache_log_dirname, corrupted_cache_log_basename)

rejected_filename_log_fname = 'rejected_halo_table_filenames.txt'
rejected_filename_log_fname = os.path.join(cache_log_dirname, rejected_filename_log_fname)

if os.path.isfile(corrupted_cache_log_fname):
    msg = ("\n\n\nThere appears to be an existing backup "
        "of the following file in your cache directory:\n\n" +
        corrupted_cache_log_fname+"\n\n"
        "This can only mean that you have run this script before in an attempt to restore your cache.\n"
        "The reason this corrupted cache is backed up is so that you do not lose a record \n"
        "of halo catalogs that were previously rejected by this script, \n"
        "but that you may have repaired in the interim.\n\n"
        "It is not permissible to run this script with this corrupted log in place, "
        "so here is how to proceed.\n"
        "Use a text editor to manually compare the "
        "corrupted and working copies of the cache log:\n\n" +
        old_cache.cache_log_fname + "\n" +
        corrupted_cache_log_fname + "\n\n"
        "For any row of the corrupted log corresponding to a halo catalog \n"
        "that you would like to be recognized in your cache,\n"
        "copy this row into a new row of " + os.path.basename(old_cache.cache_log_fname) + "\n"
        "Depending on the state of the corrupted log, "
        "you may need to enter in the metadata columns manually.\n"
        "When you have finished, delete " + os.path.basename(corrupted_cache_log_fname) + "\n"
        "after backing it up in an external location.\n"
        "Once the corrupted log has been removed from \n" + os.path.dirname(corrupted_cache_log_fname) + ",\n"
        "you can run the rebuild_halo_table_cache_log.py again.\n"
        "This script will then repeate the verification process on all entries of " +
        os.path.basename(old_cache.cache_log_fname) + "\n\n\n")
    raise HalotoolsError(msg)


def fnames_in_existing_log():
    """ If there is an existing log, try and extract a list of filenames from it.
    """
    try:
        names = [os.path.abspath(entry.fname) for entry in old_cache.log]
        existing_log_is_corrupted = False
        return names
    except:
        existing_log_is_corrupted = True
        return []


def halo_table_fnames_in_standard_cache():
    """ Walk the directory tree of all subdirectories in
    $HOME/.astropy/cache/halotools/halo_catalogs and yield the absolute path
    to any file with a .hdf5 extension.
    """
    standard_loc = os.path.join(os.path.dirname(old_cache.cache_log_fname), 'halo_catalogs')
    if os.path.exists(standard_loc):
        for path, dirlist, filelist in os.walk(standard_loc):
            for name in fnmatch.filter(filelist, '*.hdf5'):
                yield os.path.abspath(os.path.join(path, name))

num_files = len(list(halo_table_fnames_in_standard_cache()))
print("\nNumber of files detected in standard cache location = " +
    str(num_files) + "\n")
print("\nEach of these files must be opened, and the metadata will be checked, \n"
    "and consistency between the metadata and halo table will be checked.\n"
    "Estimated runtime = " + str(num_files*3) + " seconds.\n")


def fnames_in_rejected_filename_log():
    if os.path.isfile(rejected_filename_log_fname):
        with open(rejected_filename_log_fname, 'r') as f:
            for ii, line in enumerate(f):
                line = line.strip()
                if '.hdf5' == line[-5]:
                    yield os.path.abspath(str(line))


collated_names = []
names1 = fnames_in_existing_log()
collated_names.extend(names1)
names2 = list(halo_table_fnames_in_standard_cache())
collated_names.extend(names2)
names3 = list(fnames_in_rejected_filename_log())
collated_names.extend(names3)
# remove any possibly duplicated entries
potential_fnames = [str(name).strip() for name in collated_names]
potential_fnames = list(set(potential_fnames))

rejected_fnames = {}
potential_log_entries = []
for fname in potential_fnames:
    result = old_cache.determine_log_entry_from_fname(fname)
    if type(result) is HaloTableCacheLogEntry:
        potential_log_entries.append(result)
    else:
        rejected_fnames[fname] = result
potential_log_entries = list(set(potential_log_entries))

new_cache = HaloTableCache(read_log_from_standard_loc=False)
for log_entry in potential_log_entries:
    try:
        new_cache.add_entry_to_cache_log(
            log_entry, update_ascii=False)
    except:
        rejected_fnames[log_entry.fname] = log_entry._cache_safety_message

print("\nNumber of files passing verification tests = " +
    str(len(new_cache.log)) + "\n")

print("\nNumber of files that fail verification tests = " +
    str(len(rejected_fnames)) + "\n")

# We are now done with the existing rejected_fnames file.
if os.path.isfile(rejected_filename_log_fname):
    os.remove(rejected_filename_log_fname)

if old_cache_log_exists:
    os.rename(old_cache.cache_log_fname, corrupted_cache_log_fname)

if len(new_cache.log) > 0:
    new_cache._overwrite_log_ascii(new_cache.log)

    print("\n")
    print("The following log entries have been verified "
        "and added to your new cache log:\n")
    for entry in new_cache.log:
        print(entry)
    print("\n")
    print("The updated cache log is stored "
        "in the following (standard) location:\n" + new_cache.cache_log_fname)

if len(rejected_fnames) > 0:
    print("\n\nThere were some filenames that were "
        "rejected and will NOT be added to your new cache log.\n"
        "The reason for the rejection appears below each filename.")
    for ii, fname in enumerate(rejected_fnames.keys()):
        print("-----------------------------------")
        print("Rejected file #" + str(ii+1) + "\n")
        print(fname)
        print(rejected_fnames[fname])

    print("\n")

    # If there was already an existing rejected filename log, delete it
    if os.path.isfile(rejected_filename_log_fname):
        os.remove(rejected_filename_log_fname)

    with open(rejected_filename_log_fname, 'w') as f:
        names = [str(os.path.abspath(name)) for name in rejected_fnames.keys()]
        names = list(set(names))
        names.sort()
        for name in names:
            f.write(name + "\n")
    print("These rejected filenames are now stored in the following location:\n" +
        rejected_filename_log_fname + "\n")

    if old_cache_log_exists:
        print("Before running this script, you already had an existing (possibly corrupted) cache log.\n"
            "This file has been saved "
            "and is now stored in the following location:\n" +
            corrupted_cache_log_fname)
    print("\n")

print("\a\a")
