"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np

from astropy.utils.misc import NumpyRNGContext

__all__ = ('generate_locus_of_3d_points', 'generate_3d_regular_mesh',
    'generate_thin_shell_of_3d_points', 'generate_thin_cylindrical_shell_of_points')


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
        ndarrays of length npts**3 stoshell the cartesian coordinates
        of the regular grid.

    """
    x = np.linspace(0, 1, npts+1)
    y = np.linspace(0, 1, npts+1)
    z = np.linspace(0, 1, npts+1)
    delta = np.diff(x)[0]/2.
    x, y, z = np.array(np.meshgrid(x[:-1], y[:-1], z[:-1]))
    return np.vstack([x.flatten()+delta, y.flatten()+delta, z.flatten()+delta]).T


def generate_thin_shell_of_3d_points(npts, radius, xc, yc, zc, seed=None, Lbox=None):
    """ Function returns a thin shell of ``npts`` points located at a distance
    ``radius`` from the point defined by ``xc``, ``yc``, ``zc``.

    Parameters
    -----------
    npts : int
        Number of points in the output shell.

    radius : float
        Radius of the shell

    xc, yc, zc : floats
        Center of the shell

    seed : int, optional
        Random number seed used to generate the shell

    Returns
    --------
    shell : array
        npts x 3 numpy array of the points in the shell

    Examples
    --------
    >>> x0, y0, z0 = 0.05, 0.15, 0.25
    >>> radius = 0.1
    >>> shell = generate_thin_shell_of_3d_points(100, radius, x0, y0, z0)
    >>> x, y, z = shell[:,0], shell[:,1], shell[:,2]
    >>> r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
    >>> assert np.allclose(r, radius, rtol = 0.001)

    """
    with NumpyRNGContext(seed):
        x0 = np.random.uniform(-1, 1, npts)
        y0 = np.random.uniform(-1, 1, npts)
        z0 = np.random.uniform(-1, 1, npts)

    r0 = np.sqrt(x0*x0 + y0*y0 + z0*z0)
    mask = r0 == 0
    x = x0[~mask]
    y = y0[~mask]
    z = z0[~mask]
    r = r0[~mask]
    normed_x = radius*x/r
    normed_y = radius*y/r
    normed_z = radius*z/r

    if Lbox is None:
        xout = normed_x + xc
        yout = normed_y + yc
        zout = normed_z + zc
    else:
        xout = (normed_x + xc) % Lbox
        yout = (normed_y + yc) % Lbox
        zout = (normed_z + zc) % Lbox

    return np.vstack([xout, yout, zout]).T


def generate_thin_cylindrical_shell_of_points(npts, radius, half_length,
        xc, yc, zc, seed=None, Lbox=None):
    """ Function returns a thin cylindrical shell of ``npts`` points
    located at an xy-distance ``radius`` from the point
    defined by ``xc``, ``yc``, ``zc`` and distributed evenly along the
    z-direction within (zc - half_length, zc + half_length).

    Parameters
    -----------
    npts : int
        Number of points in the output shell.

    radius : float
        Radius of the cylinder

    half_length: float
        Half-length of the cylinder

    xc, yc, zc : floats
        Center of the shell

    seed : int, optional
        Random number seed used to generate the shell

    Returns
    --------
    shell : array
        npts x 3 numpy array of the points in the shell

    Examples
    --------
    >>> x0, y0, z0 = 0.05, 0.15, 0.25
    >>> radius, half_length = 0.1, 0.05
    >>> shell = generate_thin_cylindrical_shell_of_points(100, radius, half_length, x0, y0, z0)
    >>> x, y, z = shell[:,0], shell[:,1], shell[:,2]
    >>> rp = np.sqrt((x-x0)**2 + (y-y0)**2)
    >>> assert np.allclose(rp, radius, rtol = 0.001)
    >>> assert np.all(z < z0 + half_length)
    >>> assert np.all(z > z0 - half_length)

    """
    with NumpyRNGContext(seed):
        x0 = np.random.uniform(-1, 1, npts)
        y0 = np.random.uniform(-1, 1, npts)

    epsilon = 0.001
    z = np.linspace(-half_length+epsilon, half_length-epsilon, npts)

    rp0 = np.sqrt(x0*x0 + y0*y0)
    mask = rp0 == 0
    x = x0[~mask]
    y = y0[~mask]
    rp = rp0[~mask]
    normed_x = radius*x/rp
    normed_y = radius*y/rp

    if Lbox is None:
        xout = normed_x + xc
        yout = normed_y + yc
        zout = z + zc
    else:
        xout = (normed_x + xc) % Lbox
        yout = (normed_y + yc) % Lbox
        zout = (z + zc) % Lbox

    return np.vstack([xout, yout, zout]).T
