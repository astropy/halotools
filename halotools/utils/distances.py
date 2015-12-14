# -*- coding: utf-8 -*-
"""
calculate the periodic distances between two sets of points
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['distance']

import numpy as np
import collections
from astropy.table import Table
from ..custom_exceptions import HalotoolsError

def distance(x1,x2,period=None):
    """ 
    Find the Euclidean distance between x1 & x2, accounting for box periodicity.
    
    Parameters
    ----------
    x1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period
    
    x2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period.
    
    period : array_like
        Size of the simulation box along each dimension. Defines periodic boundary 
        conditioning.  Must be axis aligned.
    
    Returns
    -------
    distance : array
    
    """
    
    #process inputs
    x1 = np.asarray(x1)
    if x1.ndim ==1: x1 = np.array([x1])
    x2 = np.asarray(x2)
    if x2.ndim ==1: x2 = np.array([x2])
    if period is None:
        period = np.array([np.inf]*np.shape(x1)[-1])
    
    #check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else: k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")
    
    m = np.minimum(np.fabs(x1 - x2), period - np.fabs(x1 - x2))
    distance = np.sqrt(np.sum(m*m,axis=len(np.shape(m))-1))
    
    return distance
