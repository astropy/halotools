# -*- coding: utf-8 -*-
"""

Common functions applied to halo catalogs. 

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from warnings import warn

from astropy.table import Table 

from .group_member_generator import group_member_generator

from ..custom_exceptions import HalotoolsError 


__all__ = ('broadcast_host_halo_property', 'add_halo_hostid')

def broadcast_host_halo_property(table, halo_property_key, 
    table_is_already_sorted = False, delete_possibly_existing_column = False):
    """ Calculate a property of the host of a group system 
    and broadcast that property to all group members, 
    e.g., calculate host halo mass or group central star formation rate. 

    Parameters 
    -----------
    table : Astropy `~astropy.table.Table` 
        Table storing the halo catalog. 

    halo_property_key : string 
        Name of the column to be broadcasted to all halo members 

    table_is_already_sorted : bool, optional 
        If set to True, `group_member_generator` will skip the pre-processing 
        step of sorting the table. This improves performance, 
        but `group_member_generator` will return incorrect values 
        if the table has not been sorted properly. 
        Default is False, in which case the returned table will 
        generally be sorted in a different order than the input table. 

    delete_possibly_existing_column : bool, optional 
        If set to False, `add_halo_hostid` will raise an Exception 
        if the input table already contains a ``halo_hostid`` column.
        If True, the column will be deleted if it exists, 
        and no action will be taken if it does not exist.  
        Default is False. 

    Notes
    --------
    This function is primarily for use with Halotools-formatted halo tables. 
    For example, this function assumes that the table is sorted 
    by ['halo_upid', 'halo_hostid'], 
    and that the new column will be named ``halo_property_key_host_halo``. 
    For more general functionality, 
    use `~halotools.utils.group_member_generator` instead. 
    """

    try:
        assert type(table) == Table 
    except AssertionError:
        msg = ("\nThe input ``table`` must be an Astropy `~astropy.table.Table` object\n")
        raise HalotoolsError(msg)

    try:
        assert halo_property_key in list(table.keys())
        assert 'halo_id' in list(table.keys())
    except AssertionError:
        msg = ("\nThe input table does not the input ``halo_property_key`` = "+str(halo_property_key)+" column")
        raise HalotoolsError(msg)

    new_colname = halo_property_key + '_host_halo'
    if (new_colname in list(table.keys())) & (delete_possibly_existing_column is False):
        msg = ("\nYour input table already has an existing new_colname column name.\n"
            "If you want to overwrite this column, "
            "you must set ``delete_possibly_existing_column`` to True.\n")
        raise HalotoolsError(msg)
    elif (new_colname in list(table.keys())) & (delete_possibly_existing_column is True):
        del table[new_colname]


    if table_is_already_sorted is True:
        pass
    else:
        table.sort(['halo_hostid', 'halo_upid'])

    grouping_key = 'halo_hostid'
    requested_columns = [halo_property_key]
    group_gen = group_member_generator(table, grouping_key, requested_columns)

    result = np.zeros(len(table), dtype = table[halo_property_key].dtype)
    for first, last, member_props in group_gen:
        prop = member_props[0]
        hostprop = prop[0]
        result[first:last] = hostprop
    table[new_colname] = result



def add_halo_hostid(table, delete_possibly_existing_column = False):
    """ Function creates a new column ``halo_hostid`` for the input table. 
    For rows with ``halo_upid`` = -1, ``halo_hostid`` = ``halo_id``. Otherwise, 
    ``halo_hostid`` = ``halo_upid``. 

    Parameters 
    -----------
    table : Astropy `~astropy.table.Table` 
        Table storing the halo catalog. 

    delete_possibly_existing_column : bool, optional 
        If set to False, `add_halo_hostid` will raise an Exception 
        if the input table already contains a ``halo_hostid`` column.
        If True, the column will be deleted if it exists, 
        and no action will be taken if it does not exist.  
        Default is False. 
    """
    try:
        assert type(table) == Table 
    except AssertionError:
        msg = ("\nThe input ``table`` must be an Astropy `~astropy.table.Table` object\n")
        raise HalotoolsError(msg)

    try:
        assert 'halo_upid' in list(table.keys())
        assert 'halo_id' in list(table.keys())
    except AssertionError:
        msg = ("\nThe input table must have ``halo_upid`` and ``halo_id`` keys")
        raise HalotoolsError(msg)

    if ('halo_hostid' in list(table.keys())) & (delete_possibly_existing_column is False):
        msg = ("\nYour input table already has an existing ``halo_hostid`` column name.\n"
            "If you want to overwrite this column, "
            "you must set ``delete_possibly_existing_column`` to True.\n")
        raise HalotoolsError(msg)
    elif ('halo_hostid' in list(table.keys())) & (delete_possibly_existing_column is True):
        del table['halo_hostid']


    host_mask = table['halo_upid'] == -1
    halo_hostid = np.zeros(len(table), dtype = 'i8')
    halo_hostid[host_mask] = table['halo_id'][host_mask]
    halo_hostid[~host_mask] = table['halo_upid'][~host_mask]
    table['halo_hostid'] = halo_hostid













    



