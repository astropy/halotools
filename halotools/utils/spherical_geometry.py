"""
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__=['spherical_to_cartesian', 'chord_to_cartesian', 'sample_spherical_surface']
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

    """
    from numpy import radians, sin, cos

    rar = radians(ra)
    decr = radians(dec)

    x = cos(rar) * cos(decr)
    y = sin(rar) * cos(decr)
    z = sin(decr)

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
    """
    import numpy as np

    theta = np.asarray(theta)

    if radians is False:
        theta = np.radians(theta)

    C = 2.0*np.sin(theta/2.0)

    return C


def sample_spherical_surface(N_points):
    """
    Randomly sample the sky.

    Parameters
    ----------
    N_points : int
        number of points to sample.

    Returns
    ----------
    coords : list
        (ra,dec) coordinate pairs in degrees.
    """

    from numpy import random
    from numpy import arccos
    from math import pi

    ran1 = random.rand(N_points)  # oversample, to account for box sample
    ran2 = random.rand(N_points)  # oversample, to account for box sample

    ran1 = ran1 * 2.0 * pi  # convert to radians
    ran2 = arccos(2.0 * ran2 - 1.0) - 0.5*pi  # convert to radians

    ran1 = ran1 * 360.0 / (2.0 * pi)  # convert to degrees
    ran2 = ran2 * 360.0 / (2.0 * pi)  # convert to degrees

    ran_ra = ran1
    ran_dec = ran2

    coords = list(zip(ran_ra, ran_dec))

    return coords
