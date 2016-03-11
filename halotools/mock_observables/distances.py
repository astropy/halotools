""" Module containing pure python distance functions. 
""" 
import numpy as np 

__all__ = ('periodic_3d_distance', )
__author__ = ['Andrew Hearin']


def periodic_3d_distance(x1, y1, z1, x2, y2, z2, Lbox):
    """
    Function computes the distance between two sets of coordinates 
    with the same number of points, accounting for PBCs. 

    Parameters 
    ------------
    x1, y1, z1 : array_like 
        Length-Npts arrays storing Cartesian coordinates 

    x2, y2, z2 : array_like 
        Length-Npts arrays storing Cartesian coordinates 

    Lbox : float 
        Box length defining the periodic boundary conditions 

    Returns 
    --------
    r : array_like 
        Length-Npts array storing the 3d distance between the input 
        points, accounting for box periodicity. 
    """
    dx = np.fabs(x1 - x2)
    dx = np.fmin(dx, Lbox - dx)
    dy = np.fabs(y1 - y2)
    dy = np.fmin(dy, Lbox - dy)
    dz = np.fabs(z1 - z2)
    dz = np.fmin(dz, Lbox - dz)
    return np.sqrt(dx*dx+dy*dy+dz*dz)
