""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_formatted_halo_table_cache_log_line', 
    'get_halo_table_cache_log_header', 
    'overwrite_halo_table_cache_log', 'read_halo_table_cache_log')

import os, tempfile
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
from astropy.table import Table

import warnings
import datetime

from . import sim_defaults

from ..custom_exceptions import HalotoolsError

def get_halo_table_cache_log_fname():
    dirname = os.path.join(get_astropy_cache_dir(), 'halotools')
    return os.path.join(dirname, 'halo_table_cache_log.txt')

def get_halo_table_cache_log_header():
    return '# simname  redshift  halo_finder  version_name  fname  most_recent_use\n'

def get_formatted_halo_table_cache_log_line(simname, redshift, 
    halo_finder, version_name, fname):
    timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    formatted_line = (
        simname + '  ' + str(redshift) + '  ' + 
        halo_finder + '  ' + version_name + '  ' + fname + '  ' + timenow + '\n'
        )
    return formatted_line

def overwrite_halo_table_cache_log(new_log, **kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    with open(cache_fname, 'w') as f:
        header = get_halo_table_cache_log_header() 
        f.write(header)
        for entry in new_log:
            newline = get_formatted_halo_table_cache_log_line(
                entry['simname'], entry['redshift'], 
                entry['halo_finder'], entry['version_name'], entry['fname'])
            f.write(newline)

def read_halo_table_cache_log(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    return Table.read(cache_fname, format = 'ascii')

def update_halo_table_cache_log(simname, redshift, 
    halo_finder, version_name, fname, **kwargs):
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    pass

def verify_halo_table_cache_log_columns(log):
    correct_header = get_halo_table_cache_log_header()
    expected_key_set = set(correct_header.strip().split()[1:])
    log_key_set = set(log.keys())
    try:
        assert log_key_set == expected_key_set
    except AssertionError:
        cache_fname = get_halo_table_cache_log_fname()
        if os.path.isfile(cache_fname):
            with open(cache_fname, 'r') as f:
                actual_header = f.readline()
            if actual_header != correct_header:
                header_problem_msg = ("\nThe Halotools cache log appears to be corrupted.\n"
                    "The file " + cache_fname + "\nkeeps track of which halo catalogs"
                    "you use with Halotools.\nThe correct header for this file is "
                    + correct_header + "\nThe actual header in this file is \n" 
                    + actual_header + "\n"
                    "Your halo table cache log appears to have become corrupted.\n"
                    "Please visually inspect this file to ensure it has not been \n"
                    "accidentally overwritten. ")
            else:
                raise HalotoolsError("\nUnaddressed control flow branch. This is a bug in Halotools.\n")

        else:
            raise HalotoolsError("\nUnaddressed control flow branch. This is a bug in Halotools.\n")



def identify_halo_catalog(cache_fname, **kwargs):
    log = read_cache_memory_log(cache_fname)

    try:
        fname = kwargs['fname']
        mask = log['fname'] == fname
    except:
        pass













