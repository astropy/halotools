# -*- coding: utf-8 -*-

""" Module containing the `group_member_generator` function 
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





