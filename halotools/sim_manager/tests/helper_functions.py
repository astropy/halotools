"""
"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np

from astropy.table import Table
from astropy.table import vstack as table_vstack

from astropy.config.paths import _find_home

detected_home = _find_home()

__all__ = ('add_new_row_to_cache_log', )

dummy_cache_baseloc = os.path.join(detected_home, '.tmp_dummy_cache')
cache_basename = 'halo_table_cache_log.txt'


def get_scenario_cache_fname(scenario):
    if type(scenario) is not str:
        scenario = str(scenario)
    return os.path.join(dummy_cache_baseloc, scenario)


def add_new_row_to_cache_log(scenario,
        simname, halo_finder, redshift, version_name, **kwargs):
    if type(scenario) == int:
        scenario = str(scenario)

    try:
        new_halo_table_fname = kwargs['fname']
    except KeyError:
        new_halo_table_basename = (simname + '.' + halo_finder + '.' +
            'z' + str(np.round(redshift, 3)) + '.' + version_name + '.hdf5')
        scenario_dirname = get_scenario_cache_fname(scenario)
        new_halo_table_fname = os.path.join(scenario_dirname,
            'halo_tables', simname, halo_finder, new_halo_table_basename)

    redshift = np.round(redshift, 4)
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
