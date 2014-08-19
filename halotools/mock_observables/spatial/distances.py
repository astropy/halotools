""" 
Functions that compute distances in a mock galaxy catalog in a periodic box. 
Still largely unused in its present form, and needs to be integrated with 
the pair counter and subvolume membership methods.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

__all__=['euclidean_distance','angular_distance']

import numpy as np


def euclidean_distance(x1,x2,period=None):
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


def angular_distance(p1, p2):
    """
    Find the angular seperation between p1 & p2 in degrees.
    Parameters
    ----------
    p1 : array_like
        numpy array of 2-dimensional angular positions in degrees.
    
    p2 : array_like
        numpy array of 2-dimensional angular positions in degrees.
    
    Returns
    -------
    distance : array
        angular distance in degrees
    
    """
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    if p1.ndim == 1:
        if np.shape(p1)[0] != 2:
            raise ValueError("p1 must be a list of 2-d angular coordinates in degrees.")
    elif np.shape(p1)[-1] != 2:
        raise ValueError("p1 must be a list of 2-d angular coordinates in degrees.")
    if p2.ndim == 1:
        if np.shape(p2)[0] != 2:
            raise ValueError("p1 must be a list of 2-d angular coordinates in degrees.")
    elif np.shape(p2)[-1] != 2:
        raise ValueError("p2 must be a list of 2-d angular coordinates in degrees.")
    
    if p1.ndim == 1:
        ra1 = np.array(p1[0])
        dec1 = np.array(p1[1])
    else:
        ra1 = np.array(p1[:,0])
        dec1 = np.array(p1[:,1])
    if p2.ndim == 1:
        ra2 = np.array(p2[0])
        dec2 = np.array(p2[1])
    else:
        ra2 = np.array(p2[:,0])
        dec2 = np.array(p2[:,1])
    
    x1, y1, z1 = _spherical_to_cartesian(ra1, dec1)
    x2, y2, z2 = _spherical_to_cartesian(ra2, dec2)
    
    dot = x1*x2+y1*y2+z1*z2
    dot = np.clip(dot,-1.000000,1.000000)
    da = np.arccos(dot)
    da = np.degrees(da)
    
    return da


def _spherical_to_cartesian(ra, dec):
    """
    (Private internal function)
    Inputs in degrees. Outputs x,y,z
    """
    
    rar = np.radians(ra)
    decr = np.radians(dec)
    
    x = np.cos(rar) * np.cos(decr)
    y = np.sin(rar) * np.cos(decr)
    z = np.sin(decr)
    
    return x, y, z

