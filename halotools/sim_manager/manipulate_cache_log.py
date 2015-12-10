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




def identify_halo_catalog_fname(**kwargs):
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    log = read_cache_memory_log(cache_fname)

    try:
        fname = kwargs['fname']
        mask = log['fname'] == fname
    except:
        pass

def verify_halo_table_cache_existence(cache_fname):
    """
    """

    if not os.path.isfile(cache_fname):
        msg = ("\nThe file " + cache_fname + "\ndoes not exist. "
            "This file serves as a log for all the halo catalogs you use with Halotools.\n"
            "If you have not yet downloaded the initial halo catalog,\n"
            "you should do so now following the ``Getting Started`` instructions on "
            "http://halotools.readthedocs.org\nIf you have already taken this step,\n"
            "then your halo table cache log has been deleted,\nin which case you should"
            "execute the following script:\n"
            "halotools/scripts/auto_detect_halo_tables_in_cache.py\n")
        raise HalotoolsError(msg)

def verify_halo_table_cache_header(cache_fname):
    """
    """
    verify_halo_table_cache_existence(cache_fname)

    correct_header = get_halo_table_cache_log_header()
    with open(cache_fname, 'r') as f:
        actual_header = f.readline()

    if correct_header != actual_header:
        msg = ("\nThe file " + cache_fname + 
            "serves as a log for all the halo catalogs you use with Halotools.\n"
            "The correct header that should be in this file is \n"
            + correct_header + "\nThe actual header currently stored in this file is \n"
            + actual_header + "\nTo resolve your error, try opening the log file "
            "with a text editor and replacing the current line with the correct one.\n")
        raise HalotoolsError(msg)

def verify_halo_table_cache_log_columns(log, **kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    verify_halo_table_cache_header(cache_fname)

    correct_header = get_halo_table_cache_log_header()
    expected_key_set = set(correct_header.strip().split()[1:])
    log_key_set = set(log.keys())
    try:
        assert log_key_set == expected_key_set
    except AssertionError:
        cache_fname = get_halo_table_cache_log_fname()
        msg = ("The file " + cache_fname + 
            "\nkeeps track of the halo catalogs"
            "you use with Halotools.\n"
            "This file appears to be corrupted.\n"
            "Please visually inspect this file to ensure it has not been "
            "accidentally overwritten. \n"
            "Then store a backup of this file and execute the following script:\n"
            "halotools/scripts/auto_detect_halo_tables_in_cache.py\n"
            "If this does not resolve the error you are encountering,\n"
            "and if you have been using halo catalogs stored on some external disk \n"
            "or other non-standard location, you may try manually adding \n"
            "the appropriate lines to the cache log.\n"
            "Please contact the Halotools developers if the issue persists.\n")
        raise HalotoolsError(msg)











