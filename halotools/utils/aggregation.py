# -*- coding: utf-8 -*-
"""

Modules to support 'grouped' array calculations; aggregation operations.

"""

from __future__ import division, print_function
import numpy as np
from halotools.utils.match import match
from astropy.table import Table, Column

__all__=['add_group_property','add_members_property','group_by',\
         'binned_aggregation_group_property', 'binned_aggregation_members_property',\
         'new_members_property','new_group_property']
__author__=['Andrew Hearin', 'Duncan Campbell']

def add_group_property(members, grouping_key, function, groups, new_field_name):
    """
    Add a new group property to groups array
    
    Parameters
    ----------
    members: astropy.table.Table
        table with one row per object
    
    grouping_key: string
        key string into members which defines groups, i.e. a group ID.
    
    function: function object
        group aggregation function used to calculate group property.  function takes a 
        slice of members array corresponding to a group and returns an array of length 1.
        See tutorial for examples of aggregation functions.
    
    groups: astropy.table.Table
        Table of group properties. Must contain the grouping_key field defining 
        groups 
    
    new_field_name: string
        name of new field to be added to groups array
    
    Returns
    -------
    groups: astropy.table.Table
        Table with new group property added.
    
    Notes
    -----
    if a 'groups' table does not already exist, use 'create_groups' function.
    """
    
    if not isinstance(members,Table):
        if isinstance(members,np.ndarray):
            members = Table(members)
        else:
            raise ValueError("members parameter must be a table.")
    
    if not isinstance(groups,Table):
        if isinstance(groups,np.ndarray):
            members = Table(groups)
        else:
            raise ValueError("members parameter must be a table.")
    
    #check to see if grouping key is in groups array
    if not grouping_key in set(groups.dtype.names):
        raise ValueError("groups table must have a field that matches grouping key.")
    
    #check to see if grouping key is in members array
    if not grouping_key in set(members.dtype.names):
        raise ValueError("groups table must have a field that matches grouping key.")
    
    #define group IDs
    GroupIDs = members[grouping_key]
    
    new_prop, ID = new_group_property(members, function, None, GroupIDs = GroupIDs)

    inds1, inds2 = match(ID,groups[grouping_key])
    new_prop=new_prop[inds1]
    
    new_col = Column(name=new_field_name, data=new_prop)
    groups.add_column(new_col)
    
    return groups


def create_groups(members, grouping_key, function, new_field_name):
    """
    Create groups record array.
    
    Parameters
    ----------
    members: astropy.table.Table
        table with one row per object
    
    grouping_key: string
        key string into members which defines groups
    
    function: function object
        group aggregation function used to calculate group property.  function takes a 
        slice of members array corresponding to a group and returns an array of length 1.
        See tutorial for examples of aggregation functions.
    
    new_field_name: string
        name of new field to be included in resultant groups record array.
    
    Returns
    -------
    groups: astropy.table.Table
        table with new group property
    
    """
    
    if not isinstance(members,Table):
        if isinstance(members,np.ndarray):
            members = Table(members)
        else:
            raise ValueError("members parameter must be a table.")
    
    #check to see if grouping key is in members array
    if not grouping_key in set(members.dtype.names):
        raise ValueError("members table must have a field that matches grouping key.")
    
    #define group IDs
    GroupIDs = members[grouping_key]
    
    new_prop, ID = new_group_property(members, function, None, GroupIDs=GroupdIDs)
    
    groups = Table([ID,new_prop], names=[grouping_key,new_field_name], dtype=['<i8', new_prop.dtype.str])
    
    return groups


def add_members_property(members, group_key, new_field_name, function):
    """
    Add a new group property to members.
    
    Parameters
    ----------
    members: astropy.table.Table
        table with one row per object
    
    group_key: string
        key string into members which defines groups
    
    new_field_name: string
        name of new field to be added to members array if it does not exist
        
    function: function object
        members aggregation function used to calculate group property.  function takes a 
        slice of members array corresponding to a group and returns an array of length 1 
        or an array of equal length to the number of group members.  See tutorial for 
        examples of aggregation functions.
    
    Returns
    -------
    members: astropy.table.Table
        table with new group property appended.
    """
    
    if not isinstance(members,Table):
        if isinstance(members,np.ndarray):
            members = Table(members)
        else:
            raise ValueError("members parameter must be a table.")
    
    #check to see if keys are fields in members array
    member_keys = set(members.dtype.names)
    if group_key not in member_keys:
        raise ValueError("grouping key not in members array")

    new_prop = new_members_property(members, function, group_key)
    
    #check to see if new property field exists.
    if new_field_name in members.dtype.names:
        members[new_field_name] = new_prop
    else: # if not, append new field to members array
        new_col = Column(name=new_field_name, data=new_prop)
        members.add_column(new_col)
    
    return members


def group_by(members, keys=None, function=None, append_id_field=None):
    """
    Return group IDs of members given a grouping criteria.
    
    Parameters
    ----------
    members: astropy.table.Table
        table with one row per object
    
    keys: list, optional
        key string(s) into members which defines groups, where a group shares the same 
        value of members[key[0]], members[key[1]], etc...
    
    function: function object, optional
        function applied to members; used to group members.  e.g. function(members) may 
        return group IDs.
    
    append_id_field: string, optional
        If None, return array of length the same as members with group IDs
        otherwise, append the resulting group id as a new field in members with name given
        by append_id_field
        
    Returns
    -------
    group_ids: numpy.array
        integer array of new properties for each entry in members.  If append_id_field is
        provided, the members array is returned with a new field, append_id_field, 
        containing the group_ids.
        
    Notes
    -----
    either 'keys', or 'function' argument must be given. If 'function' is passed in,
    then the objects are grouped using the function.  Otherwise, 'keys' is used.
    """
    
    if (keys==None) & (function==None):
        raise ValueError("one of 'keys' or 'function' must be specified.")
    
    if (keys!=None) & (function!=None):
        print("using 'function' to group members, ignoring 'keys'.")
    
    if not isinstance(members,Table):
        if isinstance(members,np.ndarray):
            members = Table(members)
        else:
            raise ValueError("members parameter must be a table.")
    
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
        GroupIDs = GroupIDs[inverse_inds]
    
    if append_id_field==None: return GroupIDs #return array with group IDs
    else:
        #append new field with group IDs to members and return members
        new_col = Column(name=append_id_field, data=GroupIDs)
        members.add_column(new_col)
        return members


def binned_aggregation_group_property(members, binned_prop_key, bins, function):
    """
    Group objects by binned_prop_key and calculate a quantity by bin
    
    Parameters
    ----------
    members: astropy.table.Table
        table with one row per object
    
    binned_prop_key: string
        key string into members to bin by
    
    bins: array_like
        
    function: function object or string
        group aggregation function used to calculate group property, where groups are 
        determined by binning.  function takes a slice of members array corresponding to a
        group and returns an array of length 1.  See tutorial for examples of aggregation 
        functions.
    
    Returns
    -------
    bins, new_prop: bins, grouped by bin calculation result 
    """
    
    if not isinstance(members,Table):
        if isinstance(members,np.ndarray):
            members = Table(members)
        else:
            raise ValueError("members parameter must be a table.")
    
    GroupIDs = np.digitize(members[binned_prop_key],bins=bins)
    
    new_prop = new_group_property(members, function, None, GroupIDs=GroupIDs)[0]
    
    real_bins = [[bins[i],bins[i+1]] for i in range(0,len(bins)-1)]
    return real_bins, new_prop


def binned_aggregation_members_property(members, binned_prop_key, bins, function):
    """
    Group objects by binned_prop_key and calculate a quantity by bin
    
    Parameters
    ----------
    members: astropy.table.Table
        Table with one row per object
    
    binned_prop_key: string
        key string into members to bin by
    
    bins: array_like
        
    function: function object or string
        members aggregation function used to calculate group property, where groups are 
        determined by binning.  function takes a slice of members array corresponding to a
        group and returns an array of length 1 or an array of equal length to the number 
        of group members.  See tutorial for examples of aggregation functions.
    
    Returns
    -------
    new_prop: array of new grouped by bin properties for each member
    """
    
    if not isinstance(members,Table):
        if isinstance(members,np.ndarray):
            members = Table(members)
        else:
            raise ValueError("members parameter must be a table.")
    
    GroupIDs = np.digitize(members[binned_prop_key],bins=bins)
    
    new_prop = new_members_property(members, function, None, GroupIDs=GroupIDs)
    
    real_bins = [[bins[i],bins[i+1]] for i in range(0,len(bins)-1)]
    return real_bins, new_prop


def new_members_property(x, funcobj, grouping_key, GroupIDs=None):
    """
    Add a new group members' property.
    
    Parameters
    ----------
    x: astropy.table.Table
        table with one row per object
        
    funcobj: function
        function which operates on a record array and returns member properties
    
    grouping_key: string
        key into record array by which to group members by
    
    GroupIDs: array_like, optional
        integer array with group ID numbers.  If groupIDs not provided, grouping_key used.
    
    Returns
    -------
    result: numpy.array
    """
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
    """
    Add a new group property.
    
    Parameters
    ----------
    x: astropy.table.Table
        table with one row per object
    
    funcobj: function
        function which operates on a record array and returns a group property
    
    grouping_key: string
        key into record array by which to group members by
    
    GroupIDs: array_like, optional
        integer array with group ID numbers.  If groupIDs not provided, grouping_key used.
    
    Returns
    -------
    result, uniqueIDs
    """
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
    Identify unique rows in a record array.
    
    Parameters
    ----------
    x: numpy.recarray
        record array with one row per object
    
    Returns
    -------
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


