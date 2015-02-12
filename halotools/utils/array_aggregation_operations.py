
"""
functions to support 'grouped' array calculations
"""

def new_property(x, funcobj, grouping_key, prop_keys)
    """ Numpy-based method to compute a new property of a record array based on 
    an arbitrary input group aggregation function. 
    
    Parameters 
    ----------
    x : array_like 
        Structured numpy array
        
    funcobj : function object 
        Function used to operate on the group members. 
        The function can return either a scalar 
        that will be broadcasted amongst the members, 
        or an array with one entry per group member; 
        any_funcobj can takes a list of an arbitrary number of properties as input. 
        Function signature must be any_funcobj(x, index_array, *args), 
        where index_array is used to select the relevant group member 
        from the entire collection of galaxies, 
        and args is a list of strings that should be used 
        to access the relevant fields of x. 
        
    grouping_key : string 
        Used to define how the input x array are grouped. 
        Thus any_grouping_key must be a field name of x. 
        
    prop_keys : list of strings 
        List of field names used to access the x properties 
        that any_funcobj takes as input. 
        
    Returns 
    -------
    result : array
        The new property to attach to each element in x 
        that derives from a group-wise evaluation of any_funcobj
    """
    
    # Preform some consistency checks on the inputs
    keys = set(x.dtype.names) #turning these into sets makes it easy to check
    pkeys = set(prop_keys)
    if grouping_key not in x.dtype.names:
        rasie ValueError("grouping key not in input array")
    if not pkeys.issubset(t)
        rasie ValueError("property key(s) not a key of input array")
    
    # Initialize the output array
    result = np.zeros(len(x))

    # Calculate the array of indices that sorts the entire array by the desired key
    idx_groupsort = np.argsort(x[grouping_key])
    
    # In the sorted array, find the first entry of each new group, uniqueIDs,
    # as well as the corresponding index, uniqueID_indices
    uniqueIDs, uniqueID_indices = np.unique(
        x[grouping_key][idx_groupsort], return_index=True)
    
    # Loop over each group
    for igroup in range(len(uniqueIDs)-1):
        # Identify the indices of the sorted array corresponding to group # igroup
        idx_igroup = idx_groupsort[uniqueID_indices[igroup]:uniqueID_indices[igroup+1]]
        # Evaluate the summary statistic for this group, and store it in result
        result[idx_igroup] = any_funcobj(x, idx_igroup, *prop_keys)    
    # and now, somewhat annoyingly, repeat the above for the final group missed by the above loop
    igroup = len(uniqueIDs)-1
    idx_igroup = idx_groupsort[uniqueID_indices[igroup]:]
    result[idx_igroup] = any_funcobj(x, idx_igroup, *prop_keys)

    return result
