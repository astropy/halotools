
"""
functions to support 'grouped' array calculations; aggregation operations.  There are two 
general types of calculations supported here:
    1.) those that return a value for every member,
    2.) and those that return a value(s) for every group.
"""

from __future__ import division, print_function

__all__=['group_sum','group_mean','group_std','group_sum_of_products','group_dist','group_frac',\
         'members_sum','members_mean','members_std','members_sum_of_products','members_rank',\
         'new_group_property', 'per_group_calc']

import numpy as np
from functools import partial

###########################return new property for every group############################
def group_sum(x, grouping_key, prop_key, groups='all'):
    """
    return the aggregate sum of a property of group members per group
    """
    
    x = _get_group_subset(x,grouping_key,groups)
    
    result = per_group_calc(x, _group_sum, grouping_key, prop_key)
    return result

def group_mean(x, grouping_key, prop_key, groups='all'):
    """
    return the aggregate mean of a property of group members per group
    """
    
    x = _get_group_subset(x,grouping_key,groups)
    
    result = per_group_calc(x, _group_mean, grouping_key, prop_key)
    return result

def group_std(x, grouping_key, prop_key, groups='all'):
    """
    return the aggregate standard deviation of a property of group members per group
    """
    
    x = _get_group_subset(x,grouping_key,groups)
    
    result = per_group_calc(x, _group_std, grouping_key, prop_key)
    return result

def group_sum_of_products(x, grouping_key, prop1_key, prop2_key, groups='all'):
    """
    return the aggregate sum of products of a pair of properties of group members per group
    """
    
    x = _get_group_subset(x,grouping_key,groups)
    
    result = per_group_calc(x, _group_sum_of_products, grouping_key, [prop1_key, prop2_key])
    return result

def group_dist(x, grouping_key, prop_key, bins, groups='all'):
    """
    return the aggregate sum of products of a pair of properties of group members per group
    """
    
    x = _get_group_subset(x,grouping_key,groups)
    
    from functools import partial
    func = partial(_group_dist,bins=bins)
    
    result = per_group_calc(x, func, grouping_key, prop_key)
    return result

def group_frac(x, grouping_key, prop_key, value=1, groups='all'):
    """
    return the fraction of group members with value1
    """
    
    x = _get_group_subset(x,grouping_key,groups)
    
    from functools import partial
    func = partial(_group_frac,value=value)

    result = per_group_calc(x, func, grouping_key, prop_key)
    return result

def _get_group_subset(x,grouping_key,groups):
    
    if groups!='all':
        if not np.all(np.in1d(groups,x[grouping_key])):
            raise ValueError("all group identifiers must be in x[grouping_key]")
        else:
            #remove unnecessary entries
            keep = np.in1d(x[grouping_key],groups)
            x = x[keep]
    else: return x
    

########################return new property for every group member########################
def members_sum(x, grouping_key, prop_key):
    """
    return the aggregate sum of a property of group members
    """
    result = new_group_property(x, _members_sum, grouping_key, prop_key)
    return result

def members_mean(x, grouping_key, prop_key):
    """
    return the aggregate mean of a property of group members
    """
    result = new_group_property(x, _members_mean, grouping_key, prop_key)
    return result

def members_std(x, grouping_key, prop_key):
    """
    return the aggregate standard deviation of a property of group members
    """
    result = new_group_property(x, _members_std, grouping_key, prop_key)
    return result

def members_rank(x, grouping_key, prop_key):
    """
    return the aggregate rank order value of a property of group members
    """
    result = new_group_property(x, _members_rank, grouping_key, prop_key)
    return result

def members_sum_of_products(x, grouping_key, prop1_key, prop2_key):
    """
    return the aggregate sum of products of a pair of properties of group members
    """
    result = new_group_property(x, _members_sum_of_products, grouping_key, [prop1_key, prop2_key])
    return result
##########################################################################################


def new_group_property(x, funcobj, grouping_key, prop_keys):
    
    # Preform some consistency checks on the inputs
    keys = set(x.dtype.names) #turning these into sets makes it easy to check
    if type(prop_keys) is list: pkeys = set(prop_keys)
    else: pkeys = set([prop_keys])
    if grouping_key not in keys:
        raise ValueError("grouping key not in input array")
    if not pkeys.issubset(keys):
        raise ValueError("property key(s) not in input array")
    
    group_IDs = x[grouping_key]
    
    # Initialize the output array
    result = np.zeros(len(x))
    
    # Calculate the array of indices that sorts the entire array by the desired key
    idx_groupsort = np.argsort(group_IDs)
    
    # In the sorted array, find the first entry of each new group, uniqueIDs,
    # as well as the corresponding index, uniqueID_indices
    uniqueIDs, uniqueID_indices= np.unique(group_IDs[idx_groupsort], return_index=True)
    uniqueID_indices = np.append(uniqueID_indices,None)
    
    # Identify the indices of the sorted array corresponding to group # igroup
    for igrp_idx1, igrp_idx2 in zip(uniqueID_indices, uniqueID_indices[1:]):
        idx_igrp = idx_groupsort[igrp_idx1:igrp_idx2]
        result[idx_igrp] = funcobj(x[idx_igrp],prop_keys)
    
    return result


def per_group_calc(x, funcobj, grouping_key, prop_keys):
    
    # Preform some consistency checks on the inputs
    keys = set(x.dtype.names) #turning these into sets makes it easy to check
    if type(prop_keys) is list: pkeys = set(prop_keys)
    else: pkeys = set([prop_keys])
    if grouping_key not in keys:
        raise ValueError("grouping key not in input array")
    if not pkeys.issubset(keys):
        raise ValueError("property key(s) not in input array")
    
    group_IDs = x[grouping_key]
    
    # Calculate the array of indices that sorts the entire array by the desired key
    idx_groupsort = np.argsort(group_IDs)
    
    # In the sorted array, find the first entry of each new group, uniqueIDs,
    # as well as the corresponding index, uniqueID_indices
    uniqueIDs, uniqueID_indices= np.unique(group_IDs[idx_groupsort], return_index=True)
    uniqueID_indices = np.append(uniqueID_indices,None)
    
    # Initialize the output array
    result = np.empty(len(uniqueIDs))
    
    # Identify the indices of the sorted array corresponding to group # igroup
    i=0
    for igrp_idx1, igrp_idx2 in zip(uniqueID_indices, uniqueID_indices[1:]):
        idx_igrp = idx_groupsort[igrp_idx1:igrp_idx2]
        result[i] = funcobj(x[idx_igrp],prop_keys)
        i+=1
    
    return result


#########################functions used for 'per member' calculations#####################
def _members_mean(arr, key):
    return [arr[key].mean()]*len(arr)

def _members_std(arr, key):
    return [arr[key].std()]*len(arr)

def _members_sum(arr, key):
    return [arr[key].sum()]*len(arr)

def _members_sum_of_products(arr, keys):
    key1 = keys[0]
    key2 = keys[1]
    return [(arr[key1]*arr[key2]).sum()]*len(arr)

from scipy.stats import rankdata
def _members_rank(arr, key):
    return rankdata(-arr[key])-1

def _members_broadcast(arr, keys):
    iscen_key = keys[0]
    propkey = keys[1]
    return [(arr[iscen_key]*arr[propkey]).sum()]*len(arr)


###########################functions used for 'per group' calculations####################
def _group_frac(arr, key, value):
    type1 = (arr[key]==value)
    N1 = np.sum(type1)
    N2 = len(arr)
    return N1/(N1+N2)

def _group_dist(arr, key):
    return np.histogram(arr[key],bins=bins)[0]

def _group_mean(arr, key):
    return arr[key].mean()

def _group_std(arr, key):
    return arr[key].std()

def _group_sum(arr, key):
    return arr[key].sum()

def _group_sum_of_products(arr, keys):
    key1 = keys[0]
    key2 = keys[1]
    return (arr[key1]*arr[key2]).sum()
