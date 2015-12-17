#!/usr/bin/env python

#Duncan Campbell
#January 19, 2015
#Yale University
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['crossmatch']

import numpy as np
from ..custom_exceptions import HalotoolsError

def crossmatch(x, y):
    """
    Function determines the indices of matches in x into y
    
    Parameters 
    ----------
    x: array_like
        array to be matched

    y: array_like
        unique array to matched against

    Returns 
    -------
    match_into_y : array 
        indices in array x that return matches into array y

    matched_y : array 
        indices of array y

    Examples 
    --------
    >>> x = np.random.permutation(np.arange(0,1000))
    >>> x = np.random.permutation(x)
    >>> y = np.random.permutation(x)
    >>> match_into_y, matched_y = crossmatch(x, y)

    The returned arrays can be used as boolean masks that give 
    the entries of the input ``x`` with matching values in ``y``. 
    The following assert statement demonstrates the nature of the equality:

    >>> assert np.all(x[match_into_y] == y[matched_y])

    Notes 
    ------
    This function is tested by `~halotools.utils.tests.test_crossmatch`. 

    """
    
    #check to make sure the second list is unique
    if len(np.unique(y)) != len(y):
        msg = "\nThe second array must only contain unique entries."
        raise HalotoolsError(msg)
        
    mask = np.where(np.in1d(y,x)==True)[0]
    
    index_x = np.argsort(x)
    sorted_x = x[index_x]
    ind_x = np.searchsorted(sorted_x,y[mask])
    
    matches = index_x[ind_x]
    matched = mask
    
    return np.array(matches), matched
    