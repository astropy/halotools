#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table
from astropy.table import vstack as table_vstack

from astropy.config.paths import _find_home 

from .. import manipulate_cache_log

from ...custom_exceptions import HalotoolsError

import random, string
def randomword(*args):
    if len(args) == 1:
        length = args[0]
    else:
        length = np.random.random_integers(5, 15)
        return ''.join(random.choice(string.lowercase) for i in range(length))


### Determine whether the machine is mine
# This will be used to select tests whose 
# returned values depend on the configuration 
# of my personal cache directory files
aph_home = u'/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('TestLoadCachedHaloTableFromFname', 'create_dummy_halo_table_cache_log' )

dummy_cache_baseloc = os.path.join(detected_home, 'Desktop', 'tmp_dummy_cache')
cache_basename = 'halo_table_cache_log.txt'

def create_dummy_halo_table_cache_log(dummy_subdirname, dummy_cache_log_table):
    """
    """
    if not os.path.isdir(dummy_cache_baseloc):
        os.mkdir(dummy_cache_baseloc)
    new_dummy_cache_loc = os.path.join(dummy_cache_baseloc, dummy_subdirname)
    if not os.path.isdir(new_dummy_cache_loc):
        os.mkdir(new_dummy_cache_loc)

    dummy_cache_fname = os.path.join(new_dummy_cache_loc, cache_basename)

    manipulate_cache_log.overwrite_halo_table_cache_log(
        dummy_cache_log_table, cache_fname = dummy_cache_fname)

def add_new_cache_log_row(scenario, 
    simname, halo_finder, redshift, version_name, **kwargs):
    if type(scenario) == int:
        scenario = str(scenario)

    try:
        new_halo_table_fname = kwargs['fname']
    except KeyError:
        random_fname = randomword()
        new_halo_table_fname = os.path.join(dummy_cache_baseloc, scenario, 
            'halo_tables', simname, halo_finder, random_fname)

    new_table = Table(
        {'simname': [simname], 'halo_finder': [halo_finder], 
        'redshift': [redshift], 'version_name': [version_name], 
        'fname': [new_halo_table_fname]}
        )

    try:
        existing_table = kwargs['existing_table']
        return table_vstack([existing_table, new_table])
    except KeyError:
        return new_table

