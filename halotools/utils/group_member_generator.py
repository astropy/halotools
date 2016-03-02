# -*- coding: utf-8 -*-

""" Module containing the `group_member_generator`,  
the primary engine of the group aggregation calculations. 
"""

import numpy as np 
from astropy.table import Table 

from .array_utils import array_is_monotonic
from ..custom_exceptions import HalotoolsError

__all__ = ('add_new_table_column', )

def group_member_generator(data, grouping_key, requested_columns, 
    data_is_already_sorted=False):
    """
    Generator used to loop over grouped data and yield 
    requested properties of members of a group. 
    When running a for loop over `group_member_generator`, 
    you will be repeatedly sent arrays storing 
    properties of data entries sharing a common ``grouping_key``. 
    This enables you to perform whatever intra-group calculation 
    you wish for each iteration through the number of total groups. 
    The generator also sends you the indices of the input ``data`` 
    corresponding to the yielded group members, allowing you to 
    create new columns for your data table storing the results 
    of your intra-group calculations. 

    Common applications of `group_member_generator` include 
    subhalo analysis (e.g., calculating host halo mass) and 
    galaxy group analysis (e.g., calculating total stellar mass 
    or group-centric position). See the Examples section for further details. 

    Parameters 
    ------------
    data : Structured Numpy `~numpy.ndarray` or Astropy `~astropy.table.Table` 

    grouping_key : string 
        Name of the column that defines how the input ``data`` are grouped, 
        e.g., ``group_id`` or ``halo_hostid``. 
        The input ``data`` must be sorted such that 
        the array stored in ``data[grouping_key]`` is monotonic. 

    requested_columns : list of strings 
        List of column names that will be yielded by the generator. 
        As you loop over the generator, for every string entry in 
        ``requested_columns`` there will be an array that is yielded. 

    Returns 
    ---------
    first_idx, last_idx : int 
        These two integers provide the indices of the rows of 
        the input ``data`` yielded at each iteration. 

    group_data_list : list 
        List of arrays storing the requested group member properties. 
        There will be one element of ``group_data_list`` for every 
        element of the input ``requested_columns``. Each element is a 
        Numpy `~numpy.ndarray` with a length equal to the number of 
        members of the group. 


    Examples 
    ----------
    First let's retrieve a Halotools-formatted halo catalog storing 
    some randomly generated data. 

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    The ``halo_hostid`` is a natural grouping key for a halo table. 
    Let's use this key to broadcast the host halo mass to all members 
    of the same host halo. 
    

    """

    try:
        available_columns = data.dtype.names
    except AttributeError:
        msg = ("The input ``data`` must be an Astropy Table or Numpy Structured Array")
        raise TypeError(msg)

    try:
        assert grouping_key in available_columns
    except AssertionError:
        msg = ("Input ``grouping_key`` must be a column name of the input ``data``")
        raise KeyError(msg)

    try:
        _ = iter(requested_columns)
        for colname in requested_columns:
            assert colname in available_columns
    except TypeError:
        msg = ("\nThe input ``requested_columns`` must be an iterable sequence\n")
        raise TypeError(msg)
    except AssertionError:
        if type(requested_columns) in (str, unicode):
            msg = ("\n Your input ``requested_columns`` should be a \n"
                "list of strings, not a single string\n")
        else:
            msg = ("\nEach element of the input ``requested_columns`` must be \n"
                "an existing column name of the input ``data``.\n")
        raise KeyError(msg)


    group_id_array = np.copy(data[grouping_key])
    try:
        assert array_is_monotonic(group_id_array, strict=False) != 0
    except AssertionError:
        msg = ("Your input ``data`` must be sorted so that the ``data[grouping_key]`` is monotonic")
        raise ValueError(msg)
        
    result = np.unique(group_id_array, return_index = True, return_counts = True)
    group_ids_data, idx_groups_data, group_richness_data = result

    requested_array_list = [data[key].data for key in requested_columns]
    for igroup, host_halo_id in enumerate(group_ids_data):
        first_igroup_idx = idx_groups_data[igroup]
        last_igroup_idx = first_igroup_idx + group_richness_data[igroup]
        group_data_list = [arg[first_igroup_idx:last_igroup_idx] for arg in requested_array_list]
        yield first_igroup_idx, last_igroup_idx, group_data_list





