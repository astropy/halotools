# -*- coding: utf-8 -*-

"""
functions to assit in error estimation of mock observations.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

__all__=['jackknife_covariance_matrix','cuboid_subvolume_labels']
import numpy as np
from ..utils.array_utils import convert_to_ndarray
from ..custom_exceptions import *
from warnings import warn


def cuboid_subvolume_labels(sample, Nsub, Lbox):
    """
    return integer labels indicating which cuboid subvolume of a larger cuboid volume a 
    set of pionts occupy.
    
    Parameters
    ----------
    sample : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    Nsub : array_like
        Lenght-3 numpy array of integers indicating how many times to split the volume 
        along each dimension.  If single integer, N, is supplied, ``Nsub`` is set to 
        [N,N,N], and the volume is split along each dimension N times.  The total number 
        of subvolumes is then given by `numpy.prod(Nsub)`.
    
    Lbox : array_like
        Lenght-3 numpy array definging the legnths of the sides of the cuboid volume
        that ``sample`` occupies.  If only one number is specified, the volume is assumed 
        to cubic with sides given by np.array([Lbox]*3).
    
    Returns
    -------
    labels : numpy.array
        numpy array with integer labels in the range [1,numpy.prod(Nsub)] indicating 
        the subvolume each point in ``sample`` occupies.
    
    N_sub_vol : int
       number of subvolumes.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    Divide the volume into cubes with legnth 0.25 on a side.
    
    >>> Nsub = [4,4,4]
    >>> labels = cuboid_subvolume_labels(coords, Nsub, Lbox)
    """
    
    #process inputs and check for consistency
    sample = convert_to_ndarray(sample)
    if sample.ndim !=2:
        msg = "sample must be a legnth-N by 3 array."
        raise HalotoolsError(msg)
    elif np.shape(sample)[1] !=3:
        msg = "sample must be a legnth-N by 3 array."
        raise HalotoolsError(msg)
    if type(Nsub) is int:
        Nsub = np.array([Nsub]*3)
    else: Nsub = convert_to_ndarray(Nsub, dt=np.int)
    if np.shape(Nsub) != (3,):
        msg = "Nsub must be a length-3 array or an integer."
        raise HalotoolsError(msg)
    Lbox = convert_to_ndarray(Lbox)
    if len(Lbox) == 1:
        Lbox = np.array([Lbox[0]]*3)
    if np.shape(Lbox) != (3,):
        msg = "Lsub must be a length-3 array or a number."
        raise HalotoolsError(msg)
        
    dL = Lbox/Nsub # length of subvolumes along each dimension
    N_sub_vol = np.prod(Nsub) # total the number of subvolumes
    # create an array of unique integer IDs for each subvolume
    inds = np.arange(1, N_sub_vol+1).reshape(Nsub[0], Nsub[1], Nsub[2])
    
    #tag each particle with an integer indicating which subvolume it is in
    index = np.floor(sample/dL).astype(int)
    #take care of the case where a point falls on the boundary
    for i in range(3):
        index[:, i] = np.where(index[:, i] == Nsub[i], Nsub[i] - 1, index[:, i])
    index = inds[index[:,0],index[:,1],index[:,2]].astype(int)
    
    return index, int(N_sub_vol)


def jackknife_covariance_matrix(observations):
    """
    Calculate the covariance matrix given a sample of jackknifed "observations".
    
    Parameters
    ----------
    observations : np.ndarray
        shape (N_samples, N_observations) numpy array of observations.
    
    Returns
    -------
    cov : numpy.matrix
        covaraince matrix shape (N_observations, N_observations) with the covariance 
        between observations i,j (in the order they appear in ``observations``).
    
    Examples
    --------
    For demonstration purposes we create some random data.  Let's say we have jackknife
    samples and to estimate the errors on 15 measurements.  e.g. the two point
    correlation function in 15 radial bins.
    
    >>> observations = np.random.random((100,15))
    >>> cov = jackknife_covariance_matrix(observations)
    
    """
    
    observations =  convert_to_ndarray(observations)
    
    if observations.ndim !=2:
        msg = ("observations array must be 2-dimensional")
        raise HalotoolsError(msg)
    
    N_samples = observations.shape[0] # number of samples
    Nr = observations.shape[1] # number of observations per sample
    after_subtraction = observations - np.mean(observations, axis=0) # subtract the mean
    
    # raise a warning if N_samples < Nr
    if N_samples < Nr:
        msg = ("\n the nubumber of samples is smaller than the number of \n"
               "observations. It is recommended to increase the number \n"
               "of samples or decrease the number of observations.")
        warn(msg)
    
    cov = np.zeros((Nr,Nr)) # 2D array that stores the covariance matrix 
    for i in range(Nr):
        for j in range(Nr):
            tmp = 0.0
            for k in range(N_samples):
                tmp = tmp + after_subtraction[k,i]*after_subtraction[k,j]
                cov[i,j] = (((N_samples-1)/N_samples)*tmp)
    
    return np.matrix(cov)
