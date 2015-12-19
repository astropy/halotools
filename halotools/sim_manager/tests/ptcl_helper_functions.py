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

from .. import manipulate_ptcl_table_cache_log

from ...custom_exceptions import HalotoolsError

try:
    import h5py
except ImportError:
    warn("\nMost of the functionality of the sim_manager sub-package"
    " requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

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

__all__ = ('add_new_row_to_cache_log', 'create_dummy_ptcl_table_cache_log' )

dummy_cache_baseloc = os.path.join(detected_home, 'Desktop', 'tmp_dummy_cache')
cache_basename = 'ptcl_table_cache_log.txt'

def get_scenario_cache_fname(scenario):
    if type(scenario) is not str:
        scenario = str(scenario)
    return os.path.join(dummy_cache_baseloc, scenario)

def add_new_row_to_cache_log(scenario, 
    simname, redshift, version_name, **kwargs):
    if type(scenario) == int:
        scenario = str(scenario)

    try:
        new_ptcl_table_fname = kwargs['fname']
    except KeyError:
        new_ptcl_table_basename = (simname + '.' + 
            'z' + str(np.round(redshift, 3)) + '.' + version_name + '.hdf5')
        scenario_dirname = get_scenario_cache_fname(scenario)
        new_ptcl_table_fname = os.path.join(scenario_dirname, 
            'ptcl_tables', simname, new_ptcl_table_basename)

    redshift = np.round(redshift, 4)
    new_table = Table(
        {'simname': [simname],  
        'redshift': [redshift], 'version_name': [version_name], 
        'fname': [new_ptcl_table_fname]}
        )

    try:
        existing_table = kwargs['existing_table']
        return table_vstack([existing_table, new_table])
    except KeyError:
        return new_table

def create_ptcl_table_hdf5(cache_log_entry, **kwargs):
    try:
        num_ptcls = kwargs['num_ptcls']
    except KeyError:
        num_ptcls = 10

    try:
        Lbox = kwargs['Lbox']
    except KeyError:
        Lbox = 100.

    try:
        x = kwargs['x']
    except KeyError:
        x = np.linspace(0, 0.999*Lbox, num_ptcls)   
    try:
        y = kwargs['y']
    except KeyError:
        y = np.linspace(0, 0.999*Lbox, num_ptcls)
    try:
        z = kwargs['z']
    except KeyError:
        z = np.linspace(0, 0.999*Lbox, num_ptcls)

    table = Table({
        'x': x, 
        'y': y, 
        'z': z}
        )
    if 'omit_column' in kwargs:
        del table[kwargs['omit_column']]

    try:
        simname = kwargs['simname']
    except KeyError:
        simname = cache_log_entry['simname']
    try:
        redshift = kwargs['redshift']
    except KeyError:
        redshift = cache_log_entry['redshift']
    try:
        version_name = kwargs['version_name']
    except KeyError:
        version_name = cache_log_entry['version_name']
    try:
        fname = str(kwargs['fname'])
    except KeyError:
        fname = str(cache_log_entry['fname'])
    basename = os.path.dirname(fname)
    
    try:
        os.makedirs(basename)
    except OSError:
        pass

    table.write(fname, path='data')
    f = h5py.File(fname)
    f.attrs.create('Lbox', Lbox)
    f.attrs.create('simname', simname)
    f.attrs.create('redshift', np.round(float(redshift), 4))
    f.attrs.create('version_name', version_name)
    f.attrs.create('fname', fname)

    f.close()













