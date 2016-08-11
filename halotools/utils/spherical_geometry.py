"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

__all__ = ['spherical_to_cartesian', 'chord_to_cartesian', 'sample_spherical_surface']
__author__ = ('Duncan Campbell', )


def spherical_to_cartesian(ra, dec):
    """
    Calculate cartesian coordinates on a unit sphere given two angular coordinates.
    parameters

    Parameters
    -----------
    ra : array
        Angular coordinate in degrees

    dec : array
        Angular coordinate in degrees

    Returns
    --------
    x,y,z : sequence of arrays
        Cartesian coordinates.

    Examples
    ---------
    >>> ra, dec = 0.1, 1.5
    >>> x, y, z = spherical_to_cartesian(ra, dec)

    """

    rar = np.radians(ra)
    decr = np.radians(dec)

    x = np.cos(rar) * np.cos(decr)
    y = np.sin(rar) * np.cos(decr)
    z = np.sin(decr)

    return x, y, z


def chord_to_cartesian(theta, radians=True):
    """
    Calculate chord distance on a unit sphere given an angular distance between two
    points.

    Parameters
    -----------
    theta : array
        angular distance

    radians : bool, optional
        If True, input is interpreted as radians.
        If False, input in degrees. Default is True.

    Returns
    --------
    C : array
        chord distance

    Examples
    --------
    >>> theta = np.linspace(0, 1, 100)
    >>> chord_distance = chord_to_cartesian(theta)
    """

    theta = np.atleast_1d(theta)

    if radians is False:
        theta = np.radians(theta)

    C = 2.0*np.sin(theta/2.0)

    return C


def sample_spherical_surface(N_points, seed=None):
    """
    Randomly sample the sky.

    Parameters
    ----------
    N_points : int
        number of points to sample.

    seed : int, optional
        Random number seed permitting deterministic behavior.
        Default is None for stochastic results.

    Returns
    ----------
    coords : list
        (ra,dec) coordinate pairs in degrees.

    Examples
    ---------
    >>> angular_coords_in_degrees = sample_spherical_surface(100, seed=43)
    """

    with NumpyRNGContext(seed):
        ran1 = np.random.rand(N_points)  # oversample, to account for box sample
        ran2 = np.random.rand(N_points)  # oversample, to account for box sample

    ran1 = ran1 * 2.0 * np.pi  # convert to radians
    ran2 = np.arccos(2.0 * ran2 - 1.0) - 0.5*np.pi  # convert to radians

    ran1 = ran1 * 360.0 / (2.0 * np.pi)  # convert to degrees
    ran2 = ran2 * 360.0 / (2.0 * np.pi)  # convert to degrees

    ran_ra = ran1
    ran_dec = ran2

    coords = list(zip(ran_ra, ran_dec))

    return coords
