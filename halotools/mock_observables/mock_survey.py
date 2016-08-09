"""
create a mock redshift survey given a mock with galaxy positions and velocities.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology
from astropy.constants import c  # the speed of light


__all__ = ('ra_dec_z', )
__author__ = ['Duncan Campbell']


def ra_dec_z(x, v, cosmo=None):
    """
    Calculate the ra, dec, and redshift assuming an observer placed at (0,0,0).

    Parameters
    ----------
    x: array_like
        Npts x 3 numpy array containing 3-d positions in Mpc/h

    v: array_like
        Npts x 3 numpy array containing 3-d velocities in km/s

    cosmo : object, optional
        Instance of an Astropy `~astropy.cosmology` object.  The default is
        FlatLambdaCDM(H0=0.7, Om0=0.3)

    Returns
    -------
    ra : np.array
        right accession in radians

    dec : np.array
        declination in radians

    redshift : np.array
        "observed" redshift

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

    We do the same thing to assign random peculiar velocities:

    >>> vx,vy,vz = (np.random.random(Npts),np.random.random(Npts),np.random.random(Npts))
    >>> vels = np.vstack((vx,vy,vz)).T

    >>> from astropy.cosmology import WMAP9 as cosmo
    >>> ra, dec, redshift = ra_dec_z(coords, vels, cosmo = cosmo)
    """

    # calculate the observed redshift
    if cosmo is None:
        cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)
    c_km_s = c.to('km/s').value

    # remove h scaling from position so we can use the cosmo object
    x = x/cosmo.h

    # compute comoving distance from observer
    r = np.sqrt(x[:, 0]**2+x[:, 1]**2+x[:, 2]**2)

    # compute radial velocity
    ct = x[:, 2]/r
    st = np.sqrt(1.0 - ct**2)
    cp = x[:, 0]/np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    sp = x[:, 1]/np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    vr = v[:, 0]*st*cp + v[:, 1]*st*sp + v[:, 2]*ct

    # compute cosmological redshift and add contribution from perculiar velocity
    yy = np.arange(0, 1.0, 0.001)
    xx = cosmo.comoving_distance(yy).value
    f = interp1d(xx, yy, kind='cubic')
    z_cos = f(r)
    redshift = z_cos+(vr/c_km_s)*(1.0+z_cos)

    # calculate spherical coordinates
    theta = np.arccos(x[:, 2]/r)
    phi = np.arctan2(x[:, 1], x[:, 0])

    # convert spherical coordinates into ra,dec
    ra = phi
    dec = theta - np.pi/2.0

    return ra, dec, redshift
