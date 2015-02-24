
"""
functions to support 'grouped' array calculations; aggregation operations.  There are two 
general types of calculations supported here:
    1.) those that return a value for every member,
    2.) and those that return a value(s) for every group.
"""

from __future__ import division, print_function

__all__=['add_group_property','add_members_property','group_by',\
         'binned_aggregation_group_property', 'binned_aggregation_members_property',\
         'new_members_property','new_group_property']

import numpy as np

def add_group_property(members, group_key, new_field_name, function, keys, groups=None):
    """
    Add a new group property.
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    group_key: string
        key string into members which defines groups
        
    function: function object or string
        function used to calculate group property.
        
    keys:
       key string(s) into members that function takes as inputs
    
    groups: array_like, optional
        record array of group properties.  must contains a "GroupID" field.
    
    =======
    returns
    =======
    groups: numpy.recarray
        record array with new group property appended.
    """
    
    members = members.astype(np.recarray)
    
    if type(function) is string:
        function = _get_aggregation_function(function)
    
    #check to see if keys are fields in members array
    member_keys = set(members.dtype.names)
    if group_key not in member_keys:
        raise ValueError("grouping key not in members array")
    for key in keys:
        if key not in member_keys:
            raise ValueError("function input key '{0}' not in members array".format(key))
    
    new_prop = new_group_property(members, grouping_key, funcobj, *prop_keys)
    
    #append new field to groups array
    groups.appendfield(new_field_name)
    groups[new_field_name]=new_prop
    
    return groups


def add_members_property(members, group_key, new_field_name, function, keys):
    """
    Add a new group property to members.
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    group_key: string
        key string into members which defines groups
        
    function: function object
        function used to calculate group property
        
    keys:
       key string(s) into members that function takes as inputs
    
    =======
    returns
    =======
    members: numpy.recarray
        record array with new group property appended.
    """
    
    members = members.astype(np.recarray)
    
    if type(function) is string:
        function = _get_aggregation_function(function)
    
    #check to see if keys are fields in members array
    member_keys = set(members.dtype.names)
    if group_key not in member_keys:
        raise ValueError("grouping key not in members array")
    for key in keys:
        if key not in member_keys:
            raise ValueError("function input key '{0}' not in members array".format(key))
    
    new_prop = new_members_property(members, grouping_key, funcobj, prop_keys)
    
    #append new field to members array
    members.appendfield(new_field_name)
    members[new_field_name]=new_prop
    
    return members


def group_by(members, keys, function=None, append_as_GroupID=False):
    """
    Return group IDs of members given a grouping criteria.
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    keys: string(s)
        key string(s) into members which defines groups
    
    function: function object
        function used to calculate group property which takes keys as argument(s)
    
    append_as_GroupID: default is False
        If True, return members array with new field 'GroupID' with result
        
    =======
    returns
    =======
    new_prop: numpy.array
        array of new properties for each entry in members
    """
    
    members = members.view(np.recarray)
    
    #check to see if keys are fields in members array
    member_keys = set(members.dtype.names)
    for key in keys:
        if key not in member_keys:
            raise ValueError("function input key '{0}' not in members array".format(key))
    
    GroupIDs=np.empty((len(members),len(keys)))
    if function==None:
        for i,key in enumerate(keys):
            GroupIDs[:,i] = members[key]
        unique_rows, foo, inverse_inds = _unique_rows(GroupIDs)
        GroupIDs = np.arange(0,len(unique_rows))[inverse_inds]
        
    
    if append_as_GroupID==False: return GroupIDs
    else:
        members.appendfield('GroupID')
        members['GroupID']=GroupIDs
        return members


def binned_aggregation_group_property(members, binned_prop_key, bins, function, keys):
    """
    Group objects by binned_prop_key and calculate a quantity by bin
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    binned_prop_key: string
        key string into members to bin by
    
    bins: array_like
        
    function: function object or string
        function used to calculate group property.
        
    keys:
       key string(s) into members that function takes as inputs
    
    groups: array_like, optional
        record array of group properties.  must contains a "GroupID" field.
    
    =======
    returns
    =======
    bins, new_prop: bins, grouped by bin calculation result 
    """
    
    GroupIDs = np.digitize(members[binned_prop_key],bins=bins)
    
    new_prop = new_group_property(members, None, funcobj, *prop_keys, GroupIDs=GroupIDs)
    
    real_bins = [[bins[i],bins[i+1]] for i in range(0,len(bins)-1)]
    return real_bins, new_prop


def binned_aggregation_members_property(members, binned_prop_key, bins, function, keys):
    """
    Group objects by binned_prop_key and calculate a quantity by bin
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    binned_prop_key: string
        key string into members to bin by
    
    bins: array_like
        
    function: function object or string
        function used to calculate group property.
        
    keys:
       key string(s) into members that function takes as inputs
    
    groups: array_like, optional
        record array of group properties.  must contains a "GroupID" field.
    
    =======
    returns
    =======
    new_prop: array of new grouped by bin properties for each member
    """
    
    GroupIDs = np.digitize(members[binned_prop_key],bins=bins)
    
    new_prop = new_members_property(members, None, funcobj, prop_keys, GroupIDs=GroupIDs)
    

def _get_aggregation_function(name):
    """
    return common functions given a string argument
    """


def _unique_rows(x):
    """
    ==========
    parameters
    ==========
    x: np.ndarray
        2-dimensional array
    
    =======
    returns
    =======
    unique_rows: array
        the unique rows of A
    
    inds: array
        indicies into x[inds,:] which return array with unique_rows
    
    inverse_inds: array
        indicies into unique_rows[:,inds] which returns x
    """
    x = np.require(x, requirements='C')
    if x.ndim != 2:
        raise ValueError("array must be 2-dimensional")

    y = np.unique(x.view([('', x.dtype)]*x.shape[1]),
                  return_index=True, return_inverse=True)

    return (y[0].view(x.dtype).reshape((-1, x.shape[1]), order='C'),) + y[1:]


def new_members_property(x, funcobj, grouping_key, prop_keys, GroupIDs=None):
    
    if GroupID ==None:
        GroupIDs = x[grouping_key]
    
    # Initialize the output array
    result = np.zeros(len(x))
    
    # Calculate the array of indices that sorts the entire array by the desired key
    idx_groupsort = np.argsort(GroupIDs)
    
    # In the sorted array, find the first entry of each new group, uniqueIDs,
    # as well as the corresponding index, uniqueID_indices
    uniqueIDs, uniqueID_indices= np.unique(GroupIDs[idx_groupsort], return_index=True)
    uniqueID_indices = np.append(uniqueID_indices,None)
    
    # Identify the indices of the sorted array corresponding to group # igroup
    for igrp_idx1, igrp_idx2 in zip(uniqueID_indices, uniqueID_indices[1:]):
        idx_igrp = idx_groupsort[igrp_idx1:igrp_idx2]
        result[idx_igrp] = funcobj(x[idx_igrp],*prop_keys)
    
    return result


def new_group_property(x, funcobj, grouping_key, prop_keys, GroupIDs=None):
        
    if GroupID ==None:
        GroupIDs = x[grouping_key]
    
    # Calculate the array of indices that sorts the entire array by the desired key
    idx_groupsort = np.argsort(GroupIDs)
    
    # In the sorted array, find the first entry of each new group, uniqueIDs,
    # as well as the corresponding index, uniqueID_indices
    uniqueIDs, uniqueID_indices= np.unique(GroupIDs[idx_groupsort], return_index=True)
    uniqueID_indices = np.append(uniqueID_indices,None)
    
    # Initialize the output array
    result = np.empty(len(uniqueIDs))
    
    # Identify the indices of the sorted array corresponding to group # igroup
    i=0
    for igrp_idx1, igrp_idx2 in zip(uniqueID_indices, uniqueID_indices[1:]):
        idx_igrp = idx_groupsort[igrp_idx1:igrp_idx2]
        result[i] = funcobj(x[idx_igrp],*prop_keys)
        i+=1
    
    return result
