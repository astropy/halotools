#Duncan Campbell
#August 27, 2014
#Yale University

__all__=['spherical_to_cartesian','chord_to_cartesian','sample_spherical_surface']


def spherical_to_cartesian(ra, dec):
    """
    Calculate cartesian coordinates on a unit sphere given two angular coordinates. 
    parameters
        ra: np.array of angular coordinate in degrees
        dec: np.array of angular coordinate in degrees
    returns
        x,y,z: np.arrays cartesian coordinates 
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
    parameters
        theta: np.array of angular distance
        radians: input in radians.  Default is true  If False, input in degrees.
    returns
        C: np.array chord distance 
    """
    import numpy as np
    
    theta = np.asarray(theta)
    
    if radians==False: 
        theta = np.radians(theta)
    
    C = 2.0*np.sin(theta/2.0)
    
    return C


def sample_spherical_surface(N_points):
    """
    Randomly sample the sky.
    
    Parameters 
    ----------
    N_points: int
        number of points to sample.
    
    Returns 
    ----------
    coords: list 
        (ra,dec) coordinate pairs in degrees.
    """

    from numpy import random
    from numpy import sin, cos, arccos
    from math import pi

    ran1 = random.rand(N_points) #oversample, to account for box sample  
    ran2 = random.rand(N_points) #oversample, to account for box sample

    ran1 = ran1 * 2.0 * pi #convert to radians
    ran2 = arccos(2.0 * ran2 - 1.0) - 0.5*pi #convert to radians

    ran1 = ran1 * 360.0 / (2.0 * pi) #convert to degrees 
    ran2 = ran2 * 360.0 / (2.0 * pi) #convert to degrees

    ran_ra = ran1
    ran_dec = ran2

    coords = zip(ran_ra,ran_dec)

    return coords
