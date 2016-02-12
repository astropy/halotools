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
from .array_utils import *

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
    x1 = convert_to_ndarray(x1)
    x2 = convert_to_ndarray(x2)
    if period is None:
        period = np.array([np.inf]*k)
    else:
        period = convert_to_ndarray(period)
    
    #dimension of data
    k = np.shape(x1)[-1]
    
    if np.shape(period)==(1,):
        period = np.array([period[0]]*k)
    
    if period is None:
        period = np.array([np.inf]*k)
    else:
       period = np.array(period)
    
    #check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else: k = np.shape(x1)[-1]
    if np.shape(period)[0] != k:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")
    
    m = np.minimum(np.fabs(x1 - x2), period - np.fabs(x1 - x2))
    distance = np.sqrt(np.sum(m*m,axis=len(np.shape(m))-1))
    
    return distance
