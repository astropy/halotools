#!/usr/bin/env python

"""Command-line script to rebuild the halo table cache log"""

import argparse, os
from astropy.table import Table

from .halotools.sim_manager import manipulate_cache_log


fname_cache_log = manipulate_cache_log.get_halo_table_cache_log_fname()

def fnames_in_existing_log():
    """
    """
    try:
        manipulate_cache_log.verify_cache_log()
        existing_log = manipulate_cache_log.read_halo_table_cache_log()
        return existing_log['fname']
    except:
        return []

def halo_table_fnames_in_standard_cache():
    """
    """
    standard_loc = os.path.join(os.path.dirname(fname_cache_log), 'halo_tables')
    if os.path.exists(standard_loc):
        for path, dirlist, filelist in os.walk(dirname_halo_tables):
            for name in fnmatch.filter(filelist, '*.hdf5'):
                yield os.path.join(path, name)

potential_fnames = fnames_in_existing_log()
potential_fnames.extend(list(halo_table_fnames_in_standard_cache))
potential_fnames = list(set(potential_fnames))

def verified_fname_generator():
    """
    """
    for fname in potential_fnames:
        try:
            verified_fname = manipulate_cache_log(fname)
            yield verified_fname
        except:
            pass

verified_fnames = list(verified_fname_generator())

try:
    os.system('rm ' + fname_cache_log)
except OSError:
    pass

new_log = Table()
new_log['fname'] = verified_fnames
for entry in new_log:
    fname = entry['fname']
    print(fname)








