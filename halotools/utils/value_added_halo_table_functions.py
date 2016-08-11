"""
Common functions applied to halo catalogs.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table

from .group_member_generator import group_member_generator
from .crossmatch import crossmatch

from ..custom_exceptions import HalotoolsError

__all__ = ('broadcast_host_halo_property', 'add_halo_hostid')


def broadcast_host_halo_property(table, halo_property_key,
        delete_possibly_existing_column=False):
    """ Calculate a property of the host of a group system
    and broadcast that property to all group members,
    e.g., calculate host halo mass.

    Parameters
    -----------
    table : Astropy `~astropy.table.Table`
        Table storing the halo catalog.

    halo_property_key : string
        Name of the column to be broadcasted to all halo members

    delete_possibly_existing_column : bool, optional
        If set to False, `add_halo_hostid` will raise an Exception
        if the input table already contains a ``halo_hostid`` column.
        If True, the column will be deleted if it exists,
        and no action will be taken if it does not exist.
        Default is False.

    Notes
    --------
    This function is primarily for use with Halotools-formatted halo tables.
    In particular, this function assumes that the table has
    a ``halo_id`` and ``halo_hostid`` column.

    Examples
    ---------
    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> broadcast_host_halo_property(halocat.halo_table, 'halo_spin')

    The ``halo_table`` now has a column called ``halo_spin_host_halo``.
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
        msg = ("\nThe input table does not have the input ``halo_property_key``"" = "+str(halo_property_key)+" column")
        raise HalotoolsError(msg)

    new_colname = halo_property_key + '_host_halo'
    if (new_colname in list(table.keys())) & (delete_possibly_existing_column is False):
        msg = ("\nYour input table already has an existing new_colname column name.\n"
            "If you want to overwrite this column, "
            "you must set ``delete_possibly_existing_column`` to True.\n")
        raise HalotoolsError(msg)
    elif (new_colname in list(table.keys())) & (delete_possibly_existing_column is True):
        del table[new_colname]

    idx_halos, idx_hosts = crossmatch(table['halo_hostid'].data, table['halo_id'].data)
    table[new_colname] = np.zeros(len(table), dtype=table[halo_property_key].dtype)
    table[new_colname][idx_halos] = table[halo_property_key][idx_hosts]


def add_halo_hostid(table, delete_possibly_existing_column=False):
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

    Examples
    ---------
    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> del halocat.halo_table['halo_hostid']
    >>> add_halo_hostid(halocat.halo_table)
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
    halo_hostid = np.zeros(len(table), dtype='i8')
    halo_hostid[host_mask] = table['halo_id'][host_mask]
    halo_hostid[~host_mask] = table['halo_upid'][~host_mask]
    table['halo_hostid'] = halo_hostid
