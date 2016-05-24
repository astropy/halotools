"""
Functions to assist in error estimation of mock observations.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['jackknife_covariance_matrix','cuboid_subvolume_labels']
__author__ = ('Duncan Campbell', )

import numpy as np
from warnings import warn

from ..custom_exceptions import HalotoolsError

def cuboid_subvolume_labels(sample, Nsub, Lbox):
    """
    Return integer labels indicating which cubical subvolume of a larger cubical volume a 
    set of points occupy.
    
    Parameters
    ----------
    sample : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    Nsub : array_like
        Length-3 numpy array of integers indicating how many times to split the volume 
        along each dimension.  If single integer, N, is supplied, ``Nsub`` is set to 
        [N,N,N], and the volume is split along each dimension N times.  The total number 
        of subvolumes is then given by numpy.prod(Nsub).
    
    Lbox : array_like
        Length-3 numpy array definging the lengths of the sides of the cubical volume
        that ``sample`` occupies.  If only a single scalar is specified, the volume is assumed 
        to be a cube with side-length Lbox
    
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
    
    >>> sample = np.vstack((x,y,z)).T
    
    Divide the volume into cubes with length 0.25 on a side.
    
    >>> Nsub = [4,4,4]
    >>> labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    """
    
    #process inputs and check for consistency
    sample = np.atleast_1d(sample).astype('f8')
    try:
        assert sample.ndim == 2
        assert sample.shape[1] == 3
    except AssertionError:
        msg = ("Input ``sample`` must have shape (Npts, 3)")
        raise TypeError(msg)

    Nsub = np.atleast_1d(Nsub).astype('f4')
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0], Nsub[0], Nsub[0]])
    elif len(Nsub) != 3:
        msg = "Input ``Nsub`` must be a scalar or length-3 sequence"
        raise TypeError(msg)

    Lbox = np.atleast_1d(Lbox).astype('f8')
    if len(Lbox) == 1:
        Lbox = np.array([Lbox[0]]*3)
    elif len(Lbox) != 3:
        msg = "Input ``Lbox`` must be a scalar or length-3 sequence"
        raise TypeError(msg)
        
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
    Calculate the covariance matrix given a sample of jackknifed observations.
    
    Parameters
    ----------
    observations : np.ndarray
        shape (N_samples, N_observations) numpy array of observations.
    
    Returns
    -------
    cov : numpy.matrix
        covariance matrix shape (N_observations, N_observations) with the covariance 
        between observations i,j (in the order they appear in ``observations``).
    
    Examples
    --------
    For demonstration purposes we create some random data.  Let's say we have jackknife
    100 samples and to estimate the errors on 15 measurements, e.g. the two point
    correlation function in 15 radial bins.
    
    >>> observations = np.random.random((100,15))
    >>> cov = jackknife_covariance_matrix(observations)
    
    """
    
    observations =  np.atleast_1d(observations)
    
    if observations.ndim !=2:
        msg = ("observations array must be 2-dimensional")
        raise HalotoolsError(msg)
    
    N_samples = observations.shape[0] # number of samples
    Nr = observations.shape[1] # number of observations per sample
    after_subtraction = observations - np.mean(observations, axis=0) # subtract the mean
    
    # raise a warning if N_samples < Nr
    if N_samples < Nr:
        msg = ("\n the number of samples is smaller than the number of \n"
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
