#Duncan Campbell
#March 2015
#Yale University

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
from numpy.lib.recfunctions import append_fields

def add_group_property(members, new_field_name, function, group_key=None, groups=None):
    """
    Add a new group property.
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    new_field_name: string
        name of new field to be added to groups array
        
    function: function object or string
        function used to calculate group property.
    
    group_key: string, optional
        key string into members which defines groups
    
    groups: array_like, optional
        record array of group properties.  members and groups must contains a "GroupID" 
        field.
    
    =======
    returns
    =======
    groups: numpy.recarray
        record array with new group property appended.
    
    notes: one of either group_key, or groups must be provided.
    """
    
    members = members.view(np.recarray)
    
    if (group_key==None) & (groups==None):
        raise ValueError("one of 'group_key' or 'groups' must be privided.")
    
    if groups!=None:
        if not 'GroupID' in set(groups.dtype.names):
            raise ValueError("groups array must have a 'GroupID' field.")
        else: GroupID=groups['GroupID']
    
    if groups==None:
        GroupIDs = group_by(members, group_key)
    else:
        GroupIDs = members['GroupID']
    
    if group_key!=None:
        #check to see if keys are fields in members array
        member_keys = set(members.dtype.names)
        if group_key not in member_keys:
            raise ValueError("grouping key not in members array")
    
    new_prop, ID = new_group_property(members, function, group_key, GroupIDs = GroupIDs)
    
    print(new_prop, ID)
    
    if groups==None:
        dtype=np.dtype([('GroupID',np.int),(new_field_name,new_prop.dtype)])
        groups=np.empty((len(new_prop),),dtype=dtype)
        groups['GroupID'] = ID
        groups[new_field_name] = new_prop
    else:
        inds1, inds2 = np.match(ID,groups['GroupID'])
        new_prop=new_prop[inds]
        groups = append_fields(groups,new_field_name,new_prop)
    
    return groups


def add_members_property(members, group_key, new_field_name, function):
    """
    Add a new group property to members.
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    group_key: string
        key string into members which defines groups
    
    new_field_name: string
        name of new field to be added to members array if it does not exist
        
    function: function object
        function used to calculate group property
    
    =======
    returns
    =======
    members: numpy.recarray
        record array with new group property appended.
    """
    
    members = members.view(np.recarray)
    
    #check to see if keys are fields in members array
    member_keys = set(members.dtype.names)
    if group_key not in member_keys:
        raise ValueError("grouping key not in members array")

    new_prop = new_members_property(members, function, group_key)
    
    #check to see if new property field exists.
    if new_field_name in members.dtype.names:
        members[new_field_name] = new_prop
    else: # if not, append new field to members array
        members = append_fields(members,new_field_name,new_prop)
    
    return members


def group_by(members, keys=None, function=None, append_as_GroupID=False):
    """
    Return group IDs of members given a grouping criteria.
    
    ==========
    parameters
    ==========
    members: numpy.recarray
        record array with one row per object
    
    keys: list, optional
        key string(s) into members which defines groups
    
    function: function object, optional
        function on members used to calculate group property
    
    append_as_GroupID: bool
        If True, return members array with new field 'GroupID' with result
        If False, return array of length the same as members with group IDs
        
    =======
    returns
    =======
    new_prop: numpy.array
        array of new properties for each entry in members
        
    notes: either 'keys', or 'function' argument must be given. If keys are passed in,
    then the objects are grouped using the keys.  Otherwise, 'function' is used.
    """
    
    if (keys==None) & (function==None):
        raise ValueError("one of 'keys' or 'function' must be specified.")
    
    members = members.view(np.recarray)
    
    #check to see if keys are fields in members array
    if keys!=None:
        member_keys = set(members.dtype.names)
        for key in keys:
            if key not in member_keys:
                raise ValueError("key '{0}' not in members array".format(key))
    
    #if there is no function, assign groups by unique combination of keys
    if function==None:
        #get an array with each grouping key value per object
        GroupIDs=np.empty((len(members),len(keys)))
        for i,key in enumerate(keys):
            GroupIDs[:,i] = members[key]
        #get the unique rows in the resulting array
        unique_rows, foo, inverse_inds = _unique_rows(GroupIDs)
        #get the group ID for each object
        GroupIDs = np.arange(0,len(unique_rows))[inverse_inds]
    
    #if a function is supplied, use the function output to assign groups
    if function!=None:
        vals = function(members)
        unique_vals, inverse_inds = np.unique(vals, return_inverse=True)
        GroupIDs = np.arange(0,len(unique_vals),1).astype(int)
        GroupIDs = group_IDs[inverse_inds]
    
    if append_as_GroupID==False: return GroupIDs #return array with group IDs
    else:
        #append new field with group IDs to members and return members
        members = append_fields(members,'GroupID',GroupIDs)
        return members


def binned_aggregation_group_property(members, binned_prop_key, bins, function):
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
    
    =======
    returns
    =======
    bins, new_prop: bins, grouped by bin calculation result 
    """
    
    GroupIDs = np.digitize(members[binned_prop_key],bins=bins)
    
    new_prop = new_group_property(members, function, None, GroupIDs=GroupIDs)[0]
    
    real_bins = [[bins[i],bins[i+1]] for i in range(0,len(bins)-1)]
    return real_bins, new_prop


def binned_aggregation_members_property(members, binned_prop_key, bins, function):
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
    
    =======
    returns
    =======
    new_prop: array of new grouped by bin properties for each member
    """
    
    GroupIDs = np.digitize(members[binned_prop_key],bins=bins)
    
    new_prop = new_members_property(members, function, None, GroupIDs=GroupIDs)
    
    real_bins = [[bins[i],bins[i+1]] for i in range(0,len(bins)-1)]
    return real_bins, new_prop


def new_members_property(x, funcobj, grouping_key, GroupIDs=None):
    
    if GroupIDs == None:
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
        result[idx_igrp] = funcobj(x[idx_igrp])
    
    return result


def new_group_property(x, funcobj, grouping_key, GroupIDs=None):
        
    if GroupIDs == None:
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
        result[i] = funcobj(x[idx_igrp])
        i+=1
    
    return result, uniqueIDs


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

    y = np.unique(x.view([('',x.dtype)]*x.shape[1]),return_index=True,return_inverse=True)

    return (y[0].view(x.dtype).reshape((-1, x.shape[1]), order='C'),) + y[1:]