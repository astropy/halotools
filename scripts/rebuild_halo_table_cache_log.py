#!/usr/bin/env python

"""Command-line script to rebuild the halo table cache log"""

import argparse, os, fnmatch
from astropy.table import Table
import numpy as np

from halotools.sim_manager import manipulate_cache_log
from halotools.custom_exceptions import HalotoolsError

try:
    import h5py
except ImportError:
    msg = ("\nMust have h5py installed to use the rebuild_halo_table_cache_log script.\n")
    raise HalotoolsError(msg)

fname_cache_log = manipulate_cache_log.get_halo_table_cache_log_fname()
if os.path.isfile(fname_cache_log):
    has_existing_log = True
else:
    has_existing_log = False

cache_log_dirname = os.path.dirname(fname_cache_log)
corrupted_cache_log_fname = 'corrupted_halo_table_cache_log.txt'
corrupted_cache_log_fname = os.path.join(cache_log_dirname, corrupted_cache_log_fname)
if os.path.isfile(corrupted_cache_log_fname):
    if not os.path.isfile(fname_cache_log):
        os.system('mv '+ corrupted_cache_log_fname + ' ' + fname_cache_log)
    else:
        msg = ("\n\n\nThere appears to be an existing backup "
            "of the following file in your cache directory:\n\n"
            +corrupted_cache_log_fname+"\n\n"
            "This can only mean that you have run this script before in an attempt to restore your cache.\n"
            "The reason this corrupted cache is backed up is so that you do not lose a record \n"
            "of halo catalogs that were previously rejected by this script, \n"
            "but that you may have repaired in the interim.\n\n"
            "It is not permissible to run this script with this corrupted log in place, "
            "so here is how to proceed.\n"
            "Use a text editor to manually compare the "
            "corrupted and working copies of the cache log:\n\n"
            + fname_cache_log + "\n"
            + corrupted_cache_log_fname + "\n\n"
            "For any row of the corrupted log corresponding to a halo catalog \n"
            "that you would like to be recognized in your cache,\n"
            "copy this row into a new row of " + os.path.basename(fname_cache_log) + "\n"
            "When you have finished, delete the " + os.path.basename(corrupted_cache_log_fname) + "file \n"
            "after backing it up in an external location.\n"
            "Once the corrupted log has been removed from " + os.path.dirname(corrupted_cache_log_fname) + ",\n"
            "you can run the rebuild_halo_table_cache_log.py again.\n"
            "This script will then repeate the verification process on all entries of " 
            + os.path.basename(fname_cache_log) + "\n\n\n"
            )
        raise HalotoolsError(msg)

if os.path.exists(fname_cache_log):
    dirname = os.path.dirname(fname_cache_log)
    corrupted_cache_log_fname = os.path.join(dirname, corrupted_cache_log_fname)
    if os.path.isfile(corrupted_cache_log_fname):
        os.system('rm ' + corrupted_cache_log_fname)
    os.system('mv ' + fname_cache_log + ' ' + corrupted_cache_log_fname)


def fnames_in_existing_log():
    """
    """
    try:
        manipulate_cache_log.verify_cache_log()
        existing_log = manipulate_cache_log.read_halo_table_cache_log()
        return list(existing_log['fname'])
    except:
        return []

def halo_table_fnames_in_standard_cache():
    """
    """
    standard_loc = os.path.join(os.path.dirname(fname_cache_log), 'halo_catalogs')
    if os.path.exists(standard_loc):
        for path, dirlist, filelist in os.walk(standard_loc):
            for name in fnmatch.filter(filelist, '*.hdf5'):
                yield os.path.join(path, name)

print("\nNumber of files detected in standard cache location = " 
    + str(len(list(halo_table_fnames_in_standard_cache()))) + "\n")

potential_fnames = fnames_in_existing_log()
potential_fnames.extend(list(halo_table_fnames_in_standard_cache()))
potential_fnames = list(set(potential_fnames))

def verified_fname_generator():
    """
    """
    for fname in potential_fnames:
        try:
            verified_fname = manipulate_cache_log.verify_file_storing_unrecognized_halo_table(fname)
            yield verified_fname
        except:
            pass

def rejected_fname_generator():
    """
    """
    for fname in potential_fnames:
        try:
            verified_fname = manipulate_cache_log.verify_file_storing_unrecognized_halo_table(fname)
        except:
            yield fname

verified_fnames = list(verified_fname_generator())
print("\nNumber of files passing verification tests = " 
    + str(len(verified_fnames)) + "\n")

rejected_fnames = list(rejected_fname_generator())
print("\nNumber of files that fail verification tests = " 
    + str(len(rejected_fnames)) + "\n")


new_log = Table()
new_log['fname'] = verified_fnames
new_log['simname'] = object
new_log['halo_finder'] = object
new_log['version_name'] = object
new_log['redshift'] = 0.

for ii, entry in enumerate(new_log):
    fname = entry['fname']
    f = h5py.File(fname)
    new_log['simname'][ii] = str(f.attrs['simname'])
    new_log['halo_finder'][ii] = str(f.attrs['halo_finder'])
    new_log['version_name'][ii] = str(f.attrs['version_name'])
    new_log['redshift'][ii] = np.round(float(f.attrs['redshift']), 3)
    f.close()

if len(new_log) > 0:
    manipulate_cache_log.overwrite_halo_table_cache_log(new_log, cache_fname=fname_cache_log)

    print("\n")
    print("\n")
    print("The following filenames have been verified and will be added to your new cache log:\n")
    for entry in new_log:
        print(entry['fname'])
    print("\n")
    print("The new cache log is stored in the following location:\n" + fname_cache_log)
    print("\n")

if len(rejected_fnames) > 0:
    print("\n")
    print("The following filenames have been detected but rejected and will NOT be added to your new cache log:\n")
    for entry in rejected_fnames:
        print(entry)
    print("\n")
    if has_existing_log is True:
        print("Before running this script, you already had an existing cache log.\n"
            "This log has been saved and is now stored in the following location:\n" + corrupted_cache_log_fname)
    print("\n")
    print("To find out why a file was rejected:\n\n"
        ">>> from halotools.sim_manager import manipulate_cache_log \n"
        ">>> manipulate_cache_log.verify_file_storing_unrecognized_halo_table(fname) \n")
    print("In many cases, the reason for rejection is simple and can be easily rectified.\n"
        "For example, you may only need to add a little metadata to the .hdf5 file\n"
        "When you are done repairing any inconsistencies in the file, you can simply run this script again\n")
    print("\n")










