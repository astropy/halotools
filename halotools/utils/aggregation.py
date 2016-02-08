# -*- coding: utf-8 -*-

""" Module containing the `add_new_table_column` function 
the primary engine of the group aggregation calculations. 
"""

import numpy as np 

from astropy.table import Table 

from ..custom_exceptions import HalotoolsError

__all__ = ('add_new_table_column', )

def add_new_table_column(table, new_colname, new_coltype, grouping_key,  
    aggregation_function, colnames_needed_by_function, 
    sorting_keys = None, table_is_already_sorted = False):
    """
    Function used to add a new column to an Astropy table by 
    performing a group-wise calculation on the table elements. 

    The input ``aggregation_function`` operates on the elements 
    in each group. The value(s) returned by the function 
    is assigned to the ``new_colname`` column of the group elements. 

    Parameters 
    ------------
    table : Astropy `~astropy.table.Table` 
        Astropy table storing galaxy/halo data 

    new_colname : string 
        Name of the new column to be added to the input ``table``

    new_coltype = algebraic type
        Algebraic type used to initialize the new column. Must be 
        compatible with the usual method for declaring a Numpy data type, 
        e.g., 'f4' for float, 'f8' for double, 'i8' for long integer, etc.

    grouping_key : string 
        Column name defining how the elements of the input ``table`` 
        are arranged into groups

    aggregation_function : function 
        Function object that operates on the members of each group and 
        calculates the values stored in ``new_colname`` for the group. 
        The function must accept *num_colnames_needed* positional 
        arguments, where *num_colnames_needed* is the length of 
        the input ``colnames_needed_by_function``. 

    colnames_needed_by_function : list 
        List of strings of column names of the input ``table``. 
        The length and order of this sequence must match the 
        signature of the input ``aggregation_function``. 

    sorting_keys : list, optional 
        List of columns that defines how the input ``table`` is sorted. 
        The first element of this list must be the input ``grouping_key``. 
        Additional list elements define intra-group sorting. 
        Default is [grouping_key]. 

    table_is_already_sorted : bool, optional 
        If set to True, `add_new_table_column` will skip the pre-processing 
        step of sorting the table. This improves performance, 
        but `add_new_table_column` will return incorrect values 
        if the table has not been sorted properly. Default is False. 

    Examples 
    ----------
    First let's retrieve a Halotools-formatted halo catalog storing 
    some randomly generated data. 

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    The ``halo_hostid`` is a natural grouping key for a halo table. 
    Let's use this key to broadcast the host halo mass to all members 
    of the same host halo. 

    >>> new_colname = 'halo_mhost'
    >>> new_coltype = 'f4'
    >>> grouping_key = 'halo_hostid'
    >>> aggregation_function = lambda x: x[0] # return the value of the first group member
    >>> colnames_needed_by_function = ['halo_mvir']

    >>> sorting_keys = ['halo_hostid', 'halo_upid'] # upid = -1 for the the host halo, so that hosts occupy the first element in each grouping array
    >>> add_new_table_column(halocat.halo_table, new_colname, new_coltype, grouping_key, aggregation_function, colnames_needed_by_function, sorting_keys=sorting_keys)

    After calling `add_new_table_column` as above, 
    the ``halo_table`` has a new column named ``halo_mhost``. 

    We can also use the `add_new_table_column` to compute more complicated quantities. 
    For example, let's calculate the mean mass-weighted spin of all halo members, 
    and then broadcast the result to the members. 

    >>> new_colname = 'mass_weighted_spin'
    >>> new_coltype = 'f4'
    >>> grouping_key = 'halo_hostid'
    >>> def avg_mass_weighted_spin(mass, spin): return sum(mass*spin)/float(len(mass))
    >>> aggregation_function =  avg_mass_weighted_spin
    >>> colnames_needed_by_function = ['halo_mvir', 'halo_spin'] # In general, order is important here
    >>> sorting_keys = ['halo_hostid', 'halo_upid'] 
    >>> add_new_table_column(halocat.halo_table, new_colname, new_coltype, grouping_key, aggregation_function, colnames_needed_by_function, sorting_keys=sorting_keys)

    After calling `add_new_table_column` as above, 
    the ``halo_table`` has a new column named ``mass_weighted_spin``. 

    """
    try:
        assert type(table) == Table 
    except AssertionError:
        msg = ("\nThe input ``table`` must be an Astropy `~astropy.table.Table` object\n")
        raise HalotoolsError(msg)

    try:
        assert new_colname not in table.keys()
    except AssertionError:
        msg = ("\nThe input ``new_colname`` cannot be an existing column of the input ``table``\n")
        raise HalotoolsError(msg)

    try:
        assert grouping_key in table.keys()
    except AssertionError:
        msg = ("\nThe input ``grouping_key`` must be an existing column of the input ``table``\n")
        raise HalotoolsError(msg)

    try:
        assert callable(aggregation_function) is True
    except AssertionError:
        msg = ("\nThe input ``aggregation_function`` must be a callable function\n")
        raise HalotoolsError(msg)

    try:
        _ = iter(colnames_needed_by_function)
        for colname in colnames_needed_by_function:
            assert colname in table.keys()
    except TypeError:
        msg = ("\nThe input ``colnames_needed_by_function`` must be an iterable sequence\n")
        raise HalotoolsError(msg)
    except AssertionError:
        if type(colnames_needed_by_function) in (str, unicode):
            msg = ("\n Your input ``colnames_needed_by_function`` should be a \n"
                "list of strings, not a single string\n")
        else:
            msg = ("\nEach element of the input ``colnames_needed_by_function`` must be \n"
                "an existing column name of the input ``table``.\n")
        raise HalotoolsError(msg)

    if sorting_keys == None:
        sorting_keys = [grouping_key]

    try:
        _ = iter(sorting_keys)
        for colname in sorting_keys:
            assert colname in table.keys()
    except TypeError:
        msg = ("\nThe input ``sorting_keys`` must be an iterable sequence\n")
        raise HalotoolsError(msg)
    except AssertionError:
        if type(sorting_keys) in (str, unicode):
            msg = ("\n Your input ``sorting_keys`` should be a \n"
                "list of strings, not a single string\n")
        else:
            msg = ("\nEach element of the input ``sorting_keys`` must be \n"
                "an existing column name of the input ``table``.\n")
        raise HalotoolsError(msg)
    else:
        try:
            assert sorting_keys[0] == grouping_key
        except AssertionError:
            msg = ("\nThe first element of the input ``sorting_keys`` must be \n"
                "equal to the input ``grouping_key``\n")
            raise HalotoolsError(msg)


    if table_is_already_sorted is False:
        table.sort(sorting_keys)

    group_ids_data, idx_groups_data, group_richness_data = np.unique(
        table[grouping_key].data, 
        return_index = True, return_counts = True)

    try:
        dt = np.dtype([(new_colname, new_coltype)])
    except TypeError:
        msg = ("\nThe input ``new_coltype`` must be Numpy-compatible.\n"
            "In particular, your input must work properly with the following syntax:\n\n"
            ">>> dt = np.dtype([(new_colname, new_coltype)]) \n\n")
        raise HalotoolsError(msg)
    result = np.zeros(len(table), dtype=dt[new_colname])

    func_arglist = [table[key].data for key in colnames_needed_by_function]

    for igroup, host_halo_id in enumerate(group_ids_data):
        first_igroup_idx = idx_groups_data[igroup]
        last_igroup_idx = first_igroup_idx + group_richness_data[igroup]
        group_data_list = [arg[first_igroup_idx:last_igroup_idx] for arg in func_arglist]
        result[first_igroup_idx:last_igroup_idx] = aggregation_function(*group_data_list)

    table[new_colname] = result



