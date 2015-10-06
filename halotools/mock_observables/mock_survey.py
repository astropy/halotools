# -*- coding: utf-8 -*-

"""
create a mock redshift survey given a mock with galaxy positions and velocities.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology
from astropy.constants import c #the speed of light
##########################################################################################


__all__=['distant_observer_redshift', 'ra_dec_z']
__author__ = ['Duncan Campbell']


def distant_observer_redshift(x, v, period=None, cosmo=None):
    """
    Calculate observed redshifts using the distant observer approximation.
    
    The cosmological redshift is estimated as:
    
    z_cosmo = z*H0/c
    
    where z is the 'z' position, H0 is the Hubble constant at z=0, and c is the speed of
    light.  Note that this is an approximation
    
    Parameters
    ----------
    x: array_like
        Npts x 3 numpy array containing 3-d positions in Mpc/h units
    
    v: array_like
        Npts x 3 numpy array containing 3-d velocities of shape (N,3) in km/s
    
    period: array_like, optional
        periodic boundary conditions of simulation box
    
    Returns
    -------
    redshift: np.array
        'observed' redshift.
    """
    
    c_km_s = c.to('km/s').value
    
    #get the peculiar velocity component along the line of sight direction (z direction)
    v_los = v[:,2]
    
    #compute cosmological redshift (h=1, note that positions are in Mpc/h)
    z_cos = x[:,2]*100.0/c_km_s
    
    #redshift is combination of cosmological and peculiar velocities
    z = z_cos+(v_los/c_km_s)*(1.0+z_cos)
    
    #reflect galaxies around PBC
    if period is not None:
        z_cos_max = period[2]*100.00/c_km_s #maximum cosmological redshift
        flip = (z > z_cos_max)
        z[flip] = z[flip] - z_cos_max
        flip = (z < 0.0)
        z[flip] = z[flip] + z_cos_max
    
    return z


def ra_dec_z(x, v, cosmo=None):
    """
    Calculate ra, dec, and redshift for a mock assuming an observer placed at (0,0,0).
    
    Parameters
    ----------
    x: array_like
        Npts x 3 numpy array containing 3-d positions in Mpc/h units
    
    v: array_like
        Npts x 3 numpy array containing 3-d velocities of shape (N,3) in km/s
    
    cosmo: astropy.cosmology object, optional
        default is FlatLambdaCDM(H0=0.7, Om0=0.3)
    
    Returns
    -------
    ra: np.array
        right accession in radians
    dec: np.array
        declination in radians
    z: np.array
        redshift
    """
    
    #calculate the observed redshift
    if cosmo==None:
        cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)
    c_km_s = c.to('km/s').value
    
    #remove h scaling from position so we can use the cosmo object
    x = x/cosmo.h
    
    #compute comoving distance from observer
    r = np.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2)
    
    #compute radial velocity
    ct = x[:,2]/r
    st = np.sqrt(1.0 - ct**2)
    cp = x[:,0]/np.sqrt(x[:,0]**2 + x[:,1]**2)
    sp = x[:,1]/np.sqrt(x[:,0]**2 + x[:,1]**2)
    vr = v[:,0]*st*cp + v[:,1]*st*sp + v[:,2]*ct
    
    #compute cosmological redshift and add contribution from perculiar velocity
    yy = np.arange(0,1.0,0.001)
    xx = cosmo.comoving_distance(yy).value
    f = interp1d(xx, yy, kind='cubic')
    z_cos = f(r)
    redshift = z_cos+(vr/c_km_s)*(1.0+z_cos)

    #calculate spherical coordinates
    theta = np.arccos(x[:,2]/r)
    phi   = np.arccos(cp) #atan(y/x)
    
    #convert spherical coordinates into ra,dec
    ra  = phi
    dec = (np.pi/2.0) - theta
    
    return ra, dec, redshift
    
    
