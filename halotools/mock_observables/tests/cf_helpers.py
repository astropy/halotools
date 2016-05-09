#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 

from astropy.utils.misc import NumpyRNGContext

__all__ = ('generate_locus_of_3d_points', 'generate_3d_regular_mesh')

def generate_locus_of_3d_points(npts, xc=0.1, yc=0.1, zc=0.1, epsilon=0.001, seed=None):
    """
    Function returns a tight locus of points inside a 3d box. 

    Parameters 
    -----------
    npts : int 
        Number of desired points 

    xc, yc, zc : float 
        Midpoint value in all three dimensions 

    epsilon : float 
        Length of the box enclosing the returned locus of points

    Returns 
    ---------
    pts : array_like 
        ndarray with shape (npts, 3) of points tightly localized around 
        the point (xc, yc, zc)
    """
    with NumpyRNGContext(seed):
        x = np.random.uniform(xc - epsilon/2., xc + epsilon/2., npts)
        y = np.random.uniform(yc - epsilon/2., yc + epsilon/2., npts)
        z = np.random.uniform(zc - epsilon/2., zc + epsilon/2., npts)

    return np.vstack([x, y, z]).T

def generate_3d_regular_mesh(npts, dmin=0, dmax=1):
    """
    Function returns a regular 3d grid of npts**3 points. 

    The spacing of the grid is defined by delta = (dmax-dmin)/npts. 
    In each dimension, the first point has coordinate delta/2., 
    and the last point has coordinate dmax - delta/2.

    Parameters 
    -----------
    npts : int 
        Number of desired points per dimension. 

    dmin : float, optional 
        Minimum coordinate value of the box enclosing the grid. 
        Default is zero. 

    dmax : float, optional 
        Maximum coordinate value of the box enclosing the grid. 
        Default is one.

    Returns 
    ---------
    x, y, z : array_like 
        ndarrays of length npts**3 storing the cartesian coordinates 
        of the regular grid. 

    """
    x = np.linspace(0, 1, npts+1)
    y = np.linspace(0, 1, npts+1)
    z = np.linspace(0, 1, npts+1)
    delta = np.diff(x)[0]/2.
    x, y, z = np.array(np.meshgrid(x[:-1], y[:-1], z[:-1]))
    return np.vstack([x.flatten()+delta, y.flatten()+delta, z.flatten()+delta]).T

