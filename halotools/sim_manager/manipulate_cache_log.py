""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_formatted_cache_memory_line', 'get_cache_memory_header', 
    'write_cache_memory_log', 'read_cache_memory_log')

import os, tempfile
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
from astropy.table import Table

import warnings
import datetime

from . import sim_defaults

from ..custom_exceptions import UnsupportedSimError, CatalogTypeError

def get_cache_memory_header():
    return '# simname  redshift  halo_finder  version_name  fname  most_recent_use\n'

def get_formatted_cache_memory_line(simname, redshift, 
    halo_finder, version_name, fname):
    timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    formatted_line = (
        simname + '  ' + str(redshift) + '  ' + 
        halo_finder + '  ' + version_name + '  ' + fname + '  ' + timenow + '\n'
        )
    return formatted_line

def write_cache_memory_log(fname, table):
    """
    """
    with open(fname, 'w') as f:
        header = get_cache_memory_header() 
        f.write(header)
        for entry in table:
            newline = get_formatted_cache_memory_line(
                entry['simname'], entry['redshift'], 
                entry['halo_finder'], entry['version_name'], entry['fname'])
            f.write(newline)

def read_cache_memory_log(fname):
    """
    """
    return Table.read(fname, format = 'ascii')






