#!/usr/bin/env python

#Duncan Campbell
#January 19, 2015
#Yale University
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['match']


import numpy as np

def main():
    """
    example code calling match()
    """
    
    x = np.random.permutation(np.arange(0,1000))
    x = np.random.permutation(x)
    y = np.random.permutation(x)
    match_into_y, matched_y = match(x,y)
    
    print(np.all(x[match_into_y]==y[matched_y]))


def match(x,y):
    """
    a function that determines the indices of matches in x into y
    
    Parameters 
    ----------
    x: array_like
        array to be matched
    y: array_like
        unique array to matched against

    Returns 
    -------
    matches, matched: indices in list x that return matches into list y, indices of list y
    """
    
    #check to make sure the second list is unique
    if len(np.unique(y))!=len(y):
        "error: second array is not a unique array! returning no matches."
        return None
        
    mask = np.where(np.in1d(y,x)==True)
    
    index_x = np.argsort(x)
    sorted_x = x[index_x]
    ind_x = np.searchsorted(sorted_x,y[mask])
    
    matches = index_x[ind_x]
    matched = mask
    
    return matches, matched
    
    
if __name__ == '__main__':
    main()