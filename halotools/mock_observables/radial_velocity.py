""" Functions calculating the radial distance and radial velocity of satellites
with respect to their centrals
"""
import numpy as np


__all__ = ('radial_distance', 'radial_distance_and_velocity')


def _signed_dx(xs, xc, xperiod):
    """ Calculate the 1-d distance between xs and xc,
    accounting for periodic boundary conditions.
    The returned result is positive when xs > xc, except when PBCs are applied,
    in which case there is a sign flip.
    """
    dx_uncorrected = xs - xc
    abs_dx_uncorrected = np.abs(dx_uncorrected)
    pbc_flip_mask = abs_dx_uncorrected > xperiod/2.
    pbc_flip_sign = np.where(pbc_flip_mask, -1, 1)
    original_sign = np.sign(dx_uncorrected)
    abs_dx = np.where(pbc_flip_mask, np.abs(abs_dx_uncorrected - xperiod), abs_dx_uncorrected)
    result = pbc_flip_sign*original_sign*abs_dx
    return np.atleast_1d(result)


def radial_distance(xs, ys, zs, xc, yc, zc, period, return_signed_1d_results=False):
    """ Calculate the radial distance between the positions of a set of satellites
    and their centrals, accounting for periodic boundary conditions.

    If the coordinates ``xc, yc, zc`` of each satellite's central are not known in advance,
    these can be computed by knowing the ``halo_hostid`` of the satellites, as shown in the Examples.

    Parameters
    ----------
    xs, ys, zs : ndarrays
        Arrays of length (ngals, ) storing the xyz-coordinates of the satellites

    xc, yc, zc : ndarrays
        Arrays of length (ngals, ) storing the xyz-coordinates of the centrals

    period : float or 3-element sequence
        Length of the periodic box in each Cartesian direction.
        If a single float is given, the box will be assumed to be a periodic cube.
        If the points are not in a periodic box, set ``period`` to np.inf.

    return_signed_1d_results : bool, optional
        If set to True, the signed distances in each dimension will be returned
        in addition to the radial distance. Default is False, in which case
        only the radial distance will be returned

    Returns
    -------
    drad : ndarray
        Array of shape (ngals, ) storing the radial distances

    Examples
    --------
    For demonstration purposes we will use a fake halo catalog and compute the radial distances
    between all the satellites and their host halos.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> satmask = halocat.halo_table['halo_upid'] != -1
    >>> satellites = halocat.halo_table[satmask]

    Since Halotools catalogs do not come with the host halo position pre-computed,
    we will need to calculate this information from knowledge of the ``halo_hostid`` of each satellite.

    >>> from halotools.utils import crossmatch
    >>> idxA, idxB = crossmatch(satellites['halo_hostid'], halocat.halo_table['halo_id'])
    >>> satellites['halo_x_host_halo'] = np.nan
    >>> satellites['halo_y_host_halo'] = np.nan
    >>> satellites['halo_z_host_halo'] = np.nan
    >>> satellites['halo_x_host_halo'][idxA] = halocat.halo_table['halo_x'][idxB]
    >>> satellites['halo_y_host_halo'][idxA] = halocat.halo_table['halo_y'][idxB]
    >>> satellites['halo_z_host_halo'][idxA] = halocat.halo_table['halo_z'][idxB]

    >>> xs, ys, zs = satellites['halo_x'], satellites['halo_y'], satellites['halo_z']
    >>> xc, yc, zc = satellites['halo_x_host_halo'], satellites['halo_y_host_halo'], satellites['halo_z_host_halo']
    >>> satellites['radial_distance'] = radial_distance(xs, ys, zs, xc, yc, zc, halocat.Lbox)
    """
    try:
        xperiod, yperiod, zperiod = period
    except TypeError:
        xperiod, yperiod, zperiod = period, period, period

    dx = _signed_dx(xs, xc, xperiod)
    dy = _signed_dx(ys, yc, yperiod)
    dz = _signed_dx(zs, zc, zperiod)
    drad = np.sqrt(dx**2 + dy**2 + dz**2)
    if return_signed_1d_results:
        return drad, dx, dy, dz
    else:
        return drad


def radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs, xc, yc, zc, vxc, vyc, vzc, period):
    """ Calculate the radial distance between the positions of a set of satellites
    and their centrals, accounting for periodic boundary conditions.

    If the positions ``xc, yc, zc`` and velocities ``vxc, vyc, vzc``
    of each satellite's central are not known in advance,
    these can be computed by knowing the ``halo_hostid`` of the satellites,
    as shown in the Examples.

    Parameters
    ----------
    xs, ys, zs : ndarrays
        Arrays of length (ngals, ) storing the xyz-coordinates of the satellites

    vxs, vys, vzs : ndarrays
        Arrays of length (ngals, ) storing the xyz-velocities of the satellites

    xc, yc, zc : ndarrays
        Arrays of length (ngals, ) storing the xyz-coordinates of the centrals

    vxc, vyc, vzc : ndarrays
        Arrays of length (ngals, ) storing the xyz-velocities of the centrals

    period : float or 3-element sequence
        Length of the periodic box in each Cartesian direction.
        If a single float is given, the box will be assumed to be a periodic cube.
        If the points are not in a periodic box, set ``period`` to np.inf.

    Returns
    -------
    drad : ndarray
        Array of shape (ngals, ) storing the radial distances

    vrad : ndarray
        Array of shape (ngals, ) storing the radial velocities

    Examples
    --------
    For demonstration purposes we will use a fake halo catalog and compute the radial distances
    between all the satellites and their host halos.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> satmask = halocat.halo_table['halo_upid'] != -1
    >>> satellites = halocat.halo_table[satmask]

    Since Halotools catalogs do not come with the host halo position pre-computed,
    we will need to calculate this information from knowledge of the ``halo_hostid`` of each satellite.

    >>> from halotools.utils import crossmatch
    >>> idxA, idxB = crossmatch(satellites['halo_hostid'], halocat.halo_table['halo_id'])
    >>> satellites['halo_x_host_halo'] = np.nan
    >>> satellites['halo_y_host_halo'] = np.nan
    >>> satellites['halo_z_host_halo'] = np.nan
    >>> satellites['halo_vx_host_halo'] = np.nan
    >>> satellites['halo_vy_host_halo'] = np.nan
    >>> satellites['halo_vz_host_halo'] = np.nan
    >>> satellites['halo_x_host_halo'][idxA] = halocat.halo_table['halo_x'][idxB]
    >>> satellites['halo_y_host_halo'][idxA] = halocat.halo_table['halo_y'][idxB]
    >>> satellites['halo_z_host_halo'][idxA] = halocat.halo_table['halo_z'][idxB]
    >>> satellites['halo_vx_host_halo'][idxA] = halocat.halo_table['halo_vx'][idxB]
    >>> satellites['halo_vy_host_halo'][idxA] = halocat.halo_table['halo_vy'][idxB]
    >>> satellites['halo_vz_host_halo'][idxA] = halocat.halo_table['halo_vz'][idxB]

    >>> xs, ys, zs = satellites['halo_x'], satellites['halo_y'], satellites['halo_z']
    >>> vxs, vys, vzs = satellites['halo_vx'], satellites['halo_vy'], satellites['halo_vz']
    >>> xc, yc, zc = satellites['halo_x_host_halo'], satellites['halo_y_host_halo'], satellites['halo_z_host_halo']
    >>> vxc, vyc, vzc = satellites['halo_vx_host_halo'], satellites['halo_vy_host_halo'], satellites['halo_vz_host_halo']
    >>> drad, vrad = radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs, xc, yc, zc, vxc, vyc, vzc, halocat.Lbox)
    """
    npts = len(np.atleast_1d(xs))

    dvx = np.atleast_1d(vxs - vxc)
    dvy = np.atleast_1d(vys - vyc)
    dvz = np.atleast_1d(vzs - vzc)

    drad, dx, dy, dz = radial_distance(xs, ys, zs, xc, yc, zc, period, return_signed_1d_results=True)

    idx_nonzero_distance = drad > 0
    num_nonzero_distance = np.count_nonzero(idx_nonzero_distance)
    num_zero_distance = npts - num_nonzero_distance

    vrad = np.zeros(npts)

    if num_zero_distance > 0:
        term1a = dvx[~idx_nonzero_distance]**2
        term2a = dvy[~idx_nonzero_distance]**2
        term3a = dvz[~idx_nonzero_distance]**2
        vrad[~idx_nonzero_distance] = np.sqrt(term1a + term2a + term3a)

    if num_nonzero_distance > 0:
        term1b = dx[idx_nonzero_distance]*dvx[idx_nonzero_distance]
        term2b = dy[idx_nonzero_distance]*dvy[idx_nonzero_distance]
        term3b = dz[idx_nonzero_distance]*dvz[idx_nonzero_distance]
        numerator = term1b + term2b + term3b
        denominator = drad[idx_nonzero_distance]
        vrad[idx_nonzero_distance] = numerator/denominator

    return drad, vrad

