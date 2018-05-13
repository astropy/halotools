r"""
"""
import numpy as np
from .crossmatch import crossmatch
from .matrix_operations_3d import rotation_matrices_from_angles, rotate_vector_collection


__all__ = ('rotate_satellite_vectors', 'calculate_satellite_radial_vector',
        'reposition_satellites_from_radial_vectors')


def rotate_satellite_vectors(satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis, return_has_match=False):
    r""" Rotate an input set of `satellite_vectors` by the input `satellite_rotation_angles`
    about the axis associated with each satellite's host halo.

    Parameters
    ----------
    satellite_vectors : ndarray
        Numpy array of shape (num_sats, 3) storing a 3d vector associated with each satellite.

    satellite_hostid : ndarray
        Numpy integer array of shape (num_sats, ) storing the ID of the associated host halo.
        Entries of `satellite_hostid` without a matching entry in `host_halo_id` will not be rotated

    satellite_rotation_angles : ndarray
        Numpy array of shape (num_sats, ) storing the rotation angles in radians

    host_halo_id : ndarray
        Numpy integer array of shape (num_host_halos, ) storing
        the unique IDs of candidate host halos.

    host_halo_axis : ndarray
        Numpy array of shape (num_host_halos, 3) storing the 3d vector about which
        satellites of that host halo will be rotated.

    return_has_match : bool, optional
        Optionally return a boolean array storing whether the satellite has a matching host.
        Default is False

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (num_sats, 3) storing the rotated satellite vectors

    has_match : ndarray, optional
        Numpy boolean array of shape (num_sats, ) equals True only for those satellites
        for which there is a matching host_halo_id

    Examples
    --------
    >>> num_sats, num_host_halos = int(1e4), 100
    >>> satellite_hostid = np.random.randint(0, num_host_halos, num_sats)
    >>> satellite_rotation_angles = np.random.uniform(-np.pi/2., np.pi/2., num_sats)
    >>> satellite_vectors = np.random.uniform(-1, 1, num_sats*3).reshape((num_sats, 3))
    >>> host_halo_id = np.arange(num_host_halos)
    >>> host_halo_axis = np.random.uniform(-1, 1, num_host_halos*3).reshape((num_host_halos, 3))
    >>> rotated_vectors = rotate_satellite_vectors(satellite_vectors, satellite_hostid, satellite_rotation_angles, host_halo_id, host_halo_axis)
    """
    satellite_hostid = np.atleast_1d(satellite_hostid)
    satellite_rotation_angles = np.atleast_1d(satellite_rotation_angles)
    satellite_vectors = np.atleast_2d(satellite_vectors)
    host_halo_id = np.atleast_1d(host_halo_id)
    host_halo_axis = np.atleast_2d(host_halo_axis)

    has_match = np.isin(satellite_hostid, host_halo_id)
    satellite_rotation_angles[~has_match] = 0.

    idxA, idxB = crossmatch(satellite_hostid, host_halo_id)
    matched_host_halo_axes = host_halo_axis[idxB]
    matched_satellite_vectors = satellite_vectors[idxA]
    matched_rotation_angles = satellite_rotation_angles[idxA]

    rotation_matrices = rotation_matrices_from_angles(matched_rotation_angles, matched_host_halo_axes)
    result = rotate_vector_collection(rotation_matrices, matched_satellite_vectors)
    new_vectors = np.zeros_like(result)
    new_vectors[idxA] = result

    if return_has_match:
        return new_vectors, has_match
    else:
        return new_vectors


def calculate_satellite_radial_vector(sat_hostid, sat_x, sat_y, sat_z,
            host_halo_id, host_halo_x, host_halo_y, host_halo_z, Lbox):
    r"""
    For each satellite, calculate the radial vector pointing from the associated host halo
    to the satellite, accounting for periodic boundary conditions.

    Parameters
    ----------
    sat_hostid : ndarray
        Numpy array of integers of shape (num_sats, )

    sat_x : ndarray
        Numpy array of shape (num_sats, )

    sat_y : ndarray
        Numpy array of shape (num_sats, )

    sat_z : ndarray
        Numpy array of shape (num_sats, )

    host_halo_id : ndarray
        Numpy array of unique integers of shape (num_hosts, )

    host_halo_x : ndarray
        Numpy array of shape (num_hosts, )

    host_halo_y : ndarray
        Numpy array of shape (num_hosts, )

    host_halo_z : ndarray
        Numpy array of shape (num_hosts, )

    Lbox : scalar or 3-element tuple
        periodic boundary conditions

    Returns
    -------
    normalized_radial_vectors : ndarray
        Numpy array of shape (num_sats, 3)

        Sign convention is such that, for cases where PBCs are not operative,
        positive values correspond to satellite coordinates being larger than central coordinates,
        i.e., normalized_radial_vectors[i, 0] = xsat[i] - xcen[i], and so forth.

        When PBCs are operative, and the host halo is at the left edge of the box,
        with the satellite at the right edge, the sign of normalized_radial_vectors will be negative.

    radial_distances : ndarray
        Numpy array of shape (num_sats, )
    """
    has_match = np.isin(sat_hostid, host_halo_id)

    sat_hostid = np.atleast_1d(sat_hostid)[has_match]
    sat_x = np.atleast_1d(sat_x)[has_match]
    sat_y = np.atleast_1d(sat_y)[has_match]
    sat_z = np.atleast_1d(sat_z)[has_match]
    assert sat_hostid.shape[0] == sat_x.shape[0] == sat_y.shape[0] == sat_z.shape[0]
    num_sats = has_match.shape[0]

    host_halo_id = np.atleast_1d(host_halo_id)
    host_halo_x = np.atleast_1d(host_halo_x)
    host_halo_y = np.atleast_1d(host_halo_y)
    host_halo_z = np.atleast_1d(host_halo_z)
    assert host_halo_id.shape[0] == host_halo_x.shape[0] == host_halo_y.shape[0] == host_halo_z.shape[0]

    Lbox = np.atleast_1d(Lbox)
    if len(Lbox) == 1:
        Lbox = np.array((Lbox[0], Lbox[0], Lbox[0]))
    elif len(Lbox) != 3:
        raise ValueError("Input `Lbox` must be a scalar or 3-element sequence")

    idxA, idxB = crossmatch(sat_hostid, host_halo_id)
    matched_sat_x = sat_x[idxA]
    matched_sat_y = sat_y[idxA]
    matched_sat_z = sat_z[idxA]
    matching_host_x = host_halo_x[idxB]
    matching_host_y = host_halo_y[idxB]
    matching_host_z = host_halo_z[idxB]

    dx = _relative_positions_and_velocities(matched_sat_x, matching_host_x, period=Lbox[0])
    dy = _relative_positions_and_velocities(matched_sat_y, matching_host_y, period=Lbox[1])
    dz = _relative_positions_and_velocities(matched_sat_z, matching_host_z, period=Lbox[2])
    dr = np.sqrt(dx**2 + dy**2 + dz**2)

    xout = np.zeros(num_sats).astype(float)
    yout = np.zeros(num_sats).astype(float)
    zout = np.zeros(num_sats).astype(float)
    xout[idxA] = dx/dr
    yout[idxA] = dy/dr
    zout[idxA] = dz/dr

    normalized_radial_vectors = np.zeros((num_sats, 3))
    normalized_radial_vectors[:, 0] = xout
    normalized_radial_vectors[:, 1] = yout
    normalized_radial_vectors[:, 2] = zout

    radial_distances = np.zeros(num_sats)
    radial_distances[idxA] = dr

    return normalized_radial_vectors, radial_distances


def reposition_satellites_from_radial_vectors(satellite_position,
            orig_radial_vector, new_radial_vector, Lbox):
    r""" Given original and new host-centric coordinates for satellites, reposition the satellites
    to their new spatial coordinates, accounting for periodic boundary conditions.

    Parameters
    ----------
    satellite_position : ndarray
        Numpy array of shape (nsats, 3)

    orig_radial_vector : ndarray
        Numpy array of shape (nsats, 3)

    new_radial_vector : ndarray
        Numpy array of shape (nsats, 3)

    Lbox : scalar or 3-element tuple
        periodic boundary conditions

    Returns
    -------
    new_satellite_position : ndarray
        Numpy array of shape (nsats, 3)

    Examples
    --------
    >>> nsats = int(1e3)
    >>> Lbox = 1
    >>> satellite_position = np.random.uniform(0, Lbox, nsats*3).reshape((nsats, 3))
    >>> orig_radial_vector = np.random.uniform(-0.1, 0.1, nsats*3).reshape((nsats, 3))
    >>> new_radial_vector = np.random.uniform(-0.1, 0.1, nsats*3).reshape((nsats, 3))
    >>> new_satellite_position = reposition_satellites_from_radial_vectors(satellite_position, orig_radial_vector, new_radial_vector, Lbox)
    """
    xc = satellite_position[:, 0] - orig_radial_vector[:, 0]
    yc = satellite_position[:, 1] - orig_radial_vector[:, 1]
    zc = satellite_position[:, 2] - orig_radial_vector[:, 2]

    xs_new = xc + new_radial_vector[:, 0]
    ys_new = yc + new_radial_vector[:, 1]
    zs_new = zc + new_radial_vector[:, 2]

    Lbox = np.atleast_1d(Lbox)
    if len(Lbox) == 1:
        Lbox = np.array((Lbox[0], Lbox[0], Lbox[0]))
    elif len(Lbox) != 3:
        raise ValueError("Input `Lbox` must be a scalar or 3-element sequence")

    xs_new = xs_new % Lbox[0]
    ys_new = ys_new % Lbox[1]
    zs_new = zs_new % Lbox[2]

    new_satellite_position = np.vstack((xs_new, ys_new, zs_new)).T
    return new_satellite_position


def _sign_pbc(x1, x2, period=None, equality_fill_val=0., return_pbc_correction=False):
    r""" Return the sign of the unit vector pointing from x2 towards x1,
    that is, the sign of (x1 - x2), accounting for periodic boundary conditions.

    If x1 > x2, returns 1. If x1 < x2, returns -1. If x1 == x2, returns equality_fill_val.

    Parameters
    ----------
    x1 : array
        1-d array of length *Npts*.
        If period is not None, all values must be contained in [0, Lbox)

    x2 : array
        1-d array of length *Npts*.
        If period is not None, all values must be contained in [0, Lbox)

    period : float, optional
        Size of the periodic box. Default is None for non-periodic case.

    equality_fill_val : float, optional
        Value to return for cases where x1 == x2. Default is 0.

    return_pbc_correction : bool, optional
        If True, the `sign_pbc` function will additionally return a
        length *Npts* boolean array storing whether or not the input
        points had a PBC correction applied. Default is False.

    Returns
    -------
    sgn : array
        1-d array of length *Npts*.

    Examples
    --------
    >>> Lbox = 250.0
    >>> x1 = 1.
    >>> x2 = 249.
    >>> result = _sign_pbc(x1, x2, period=Lbox)
    >>> assert result == 1

    >>> result = _sign_pbc(x1, x2, period=None)
    >>> assert result == -1

    >>> npts = 100
    >>> x1 = np.random.uniform(0, Lbox, npts)
    >>> x2 = np.random.uniform(0, Lbox, npts)
    >>> result = _sign_pbc(x1, x2, period=Lbox)
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    result = np.sign(x1 - x2)

    if period is not None:
        try:
            assert np.all(x1 >= 0)
            assert np.all(x2 >= 0)
            assert np.all(x1 < period)
            assert np.all(x2 < period)
        except AssertionError:
            msg = "If period is not None, all values of x and y must be between [0, period)"
            raise ValueError(msg)

        d = np.abs(x1-x2)
        pbc_correction = np.sign(period/2. - d)
        result = pbc_correction*result

    if equality_fill_val != 0:
        result = np.where(result == 0, equality_fill_val, result)

    if return_pbc_correction:
        return result, pbc_correction
    else:
        return result


def _relative_positions_and_velocities(x1, x2, period=None, **kwargs):
    r""" Return the vector pointing from x2 towards x1,
    that is, x1 - x2, accounting for periodic boundary conditions.

    If keyword arguments ``v1`` and ``v2`` are passed in,
    additionally return the velocity ``v1`` with respect to ``v2``, with sign convention
    such that positive (negative) values correspond to receding (approaching) points.

    Parameters
    -----------
    x1 : array
        1-d array of length *Npts*.
        If period is not None, all values must be contained in [0, Lbox)

    x2 : array
        1-d array of length *Npts*.
        If period is not None, all values must be contained in [0, Lbox)

    period : float, optional
        Size of the periodic box. Default is None for non-periodic case.

    Returns
    --------
    xrel : array
        1-d array of length *Npts* storing x1 - x2.
        If *x1 > x2* and abs(*x1* - *x2*) > period/2, the sign of *d* will be negative.

    vrel : array, optional
        1-d array of length *Npts* storing v1 relative to v2.
        Only returned if ``v1`` and ``v2`` are passed in.

    Examples
    --------
    >>> Lbox = 250.0
    >>> x1 = 1.
    >>> x2 = 249.
    >>> result = _relative_positions_and_velocities(x1, x2, period=Lbox)
    >>> assert np.isclose(result, 2)

    >>> result = _relative_positions_and_velocities(x1, x2, period=None)
    >>> assert np.isclose(result, -248)

    >>> npts = 100
    >>> x1 = np.random.uniform(0, Lbox, npts)
    >>> x2 = np.random.uniform(0, Lbox, npts)
    >>> result = _relative_positions_and_velocities(x1, x2, period=Lbox)

    Now let's frame this result in terms of a physically motivated example.
    Suppose we have a central galaxy with position *xc* and velocity *vc*,
    and a satellite galaxy with position *xs* and velocity *vs*.
    We can calculate the vector pointing from the central to the satellite,
    as well as the satellites's host-centric velocity:

    >>> xcen, vcen = 249.9, 100
    >>> xsat, vsat = 0.1, -300
    >>> xrel, vrel = _relative_positions_and_velocities(xsat, xcen, v1=vsat, v2=vcen, period=Lbox)
    >>> assert np.isclose(xrel, +0.2)
    >>> assert np.isclose(vrel, -400)

    >>> xcen, vcen = 0.1, 100
    >>> xsat, vsat = 249.9, -300
    >>> xrel, vrel = _relative_positions_and_velocities(xsat, xcen, v1=vsat, v2=vcen, period=Lbox)
    >>> assert np.isclose(xrel, -0.2)
    >>> assert np.isclose(vrel, +400)

    """
    s = _sign_pbc(x1, x2, period=period, equality_fill_val=1.)
    absd = np.abs(x1 - x2)
    if period is None:
        xrel = s*absd
    else:
        xrel = s*np.where(absd > period/2., period - absd, absd)

    try:
        v1 = kwargs['v1']
        v2 = kwargs['v2']
        return xrel, s*(v1-v2)
    except KeyError:
        return xrel

