r""" Common functions used when analyzing catalogs of galaxies/halos.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.stats import binned_statistic

from ..empirical_models import enforce_periodicity_of_box
from ..sim_manager.sim_defaults import default_cosmology, default_redshift

from ..custom_exceptions import HalotoolsError

__all__ = ('mean_y_vs_x', 'return_xyz_formatted_array', 'cuboid_subvolume_labels',
    'relative_positions_and_velocities', 'sign_pbc', 'apply_zspace_distortion')
__author__ = ['Andrew Hearin']


def mean_y_vs_x(x, y, error_estimator='error_on_mean', **kwargs):
    r"""
    Estimate the mean value of the property *y* as a function of *x*
    for an input sample of galaxies/halos,
    optionally returning an error estimate.

    The `mean_y_vs_x` function is just a convenience wrapper
    around `scipy.stats.binned_statistic` and `np.histogram`.

    See also :ref:`galaxy_catalog_analysis_tutorial1`.

    Parameters
    -----------
    x : array_like
        Array storing values of the independent variable of the sample.

    y : array_like
        Array storing values of the dependent variable of the sample.

    bins : array_like, optional
        Bins of the input *x*.
        Defaults are set by `scipy.stats.binned_statistic`.

    error_estimator : string, optional
        If set to ``error_on_mean``, function will also return an array storing
        :math:`\sigma_{y}/\sqrt{N}`, where :math:`\sigma_{y}` is the
        standard deviation of *y* in the bin
        and :math:`\sqrt{N}` is the counts in each bin.

        If set to ``variance``, function will also return an array storing
        :math:`\sigma_{y}`.

        Default is ``error_on_mean``

    Returns
    ----------
    bin_midpoints : array_like
        Midpoints of the *x*-bins.

    mean : array_like
        Mean of *y* estimated in bins

    err : array_like
        Error on *y* estimated in bins

    Examples
    ---------
    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> halos = halocat.halo_table
    >>> halo_mass, mean_spin, err = mean_y_vs_x(halos['halo_mvir'], halos['halo_spin'])

    See also
    ---------
    :ref:`galaxy_catalog_analysis_tutorial1`

    """
    try:
        assert error_estimator in ('error_on_mean', 'variance')
    except AssertionError:
        msg = ("\nInput ``error_estimator`` must be either "
            "``error_on_mean`` or ``variance``\n")
        raise HalotoolsError(msg)

    modified_kwargs = {key: kwargs[key] for key in kwargs if key != 'error_estimator'}
    result = binned_statistic(x, y, statistic='mean', **modified_kwargs)
    mean, bin_edges, binnumber = result
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1])/2.

    modified_kwargs['bins'] = bin_edges

    result = binned_statistic(x, y, statistic=np.std, **modified_kwargs)
    variance, _, _ = result

    if error_estimator == 'variance':
        err = variance
    else:
        counts = np.histogram(x, bins=bin_edges)
        err = variance/np.sqrt(counts[0])

    return bin_midpoints, mean, err


def return_xyz_formatted_array(x, y, z, period=np.inf,
        cosmology=default_cosmology, redshift=default_redshift, **kwargs):
    r""" Returns a Numpy array of shape *(Npts, 3)* storing the
    xyz-positions in the format used throughout
    the `~halotools.mock_observables` package, optionally applying redshift-space
    distortions according to the input ``velocity``, ``redshift`` and ``cosmology``.

    See :ref:`mock_obs_pos_formatting` for a tutorial.

    Parameters
    -----------
    x, y, z : sequence of length-Npts arrays
        Comoving units of Mpc assuming h=1, as throughout Halotools.

    velocity : array, optional
        Length-Npts array of velocities in *physical* units of km/s
        used to apply peculiar velocity distortions, e.g.,
        :math:`z_{\rm dist} = z_{\rm true} + v_{\rm z}/aH`,
        where *a* and *H* are the scale factor and Hubble expansion rate
        evaluated at the input ``redshift``.

        If ``velocity`` argument is passed,
        ``velocity_distortion_dimension`` must also be passed.

    velocity_distortion_dimension : string, optional
        If set to ``'x'``, ``'y'`` or ``'z'``,
        the requested dimension in the returned ``pos`` array
        will be distorted due to peculiar motion.
        For example, if ``velocity_distortion_dimension`` is ``z``,
        then ``pos`` can be treated as physically observed
        galaxy positions under the distant-observer approximation.
        Default is no distortions.

    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology to assume when applying redshift-space distortions,
        e.g., the cosmology of the simulation.
        Default is set in `sim_manager.sim_defaults`.

    redshift : float, optional
        Redshift of the mock galaxy sample,
        e.g., the redshift of the simulation snapshot.
        Default is set in `sim_manager.sim_defaults`.

    mask : array_like, optional
        Boolean mask that can be used to select the positions
        of a subcollection of the galaxies stored in the ``galaxy_table``.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar,
        period is assumed to be the same in all Cartesian directions.
        If period is not np.inf, then after applying peculiar velocity distortions
        the new coordinates will be remapped into the periodic box.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    Returns
    --------
    pos : array_like
        Numpy array with shape *(Npts, 3)* with units of comoving Mpc/h.

    Examples
    ---------
    >>> npts = 100
    >>> Lbox = 250.
    >>> x = np.random.uniform(0, Lbox, npts)
    >>> y = np.random.uniform(0, Lbox, npts)
    >>> z = np.random.uniform(0, Lbox, npts)
    >>> pos = return_xyz_formatted_array(x, y, z, period=Lbox)

    Now we will define an array of random velocities that we will use
    to apply z-space distortions to the z-dimension, assuming the mock galaxy
    sample is at the default redshift. For our random velocities
    we'll assume the values are drawn from a Gaussian centered at zero
    using `numpy.random.normal`.

    >>> velocity = np.random.normal(loc=0, scale=100, size=npts)
    >>> pos = return_xyz_formatted_array(x, y, z, period=Lbox, velocity=velocity, velocity_distortion_dimension='z')

    If we wanted to introduce redshift-space distortions at some higher redshift:

    >>> pos = return_xyz_formatted_array(x, y, z, period=Lbox, velocity=velocity, velocity_distortion_dimension='z', redshift=1.5)

    Notes
    -----
    See :ref:`zspace_distortion_derivation`.

    """
    period = np.atleast_1d(period)
    if len(period) == 1:
        period = np.repeat(period, 3)
    elif len(period) == 3:
        pass
    else:
        msg = "Input ``period`` must be a single float or a 3-element sequence"
        raise ValueError(msg)

    posdict = {'x': np.copy(x), 'y': np.copy(y), 'z': np.copy(z)}
    period_dict = {'x': period[0], 'y': period[1], 'z': period[2]}

    a = 'velocity_distortion_dimension' in list(kwargs.keys())
    b = 'velocity' in list(kwargs.keys())
    if bool(a+b) is True:
        if bool(a*b) is False:
            msg = ("You must either both or none of the following keyword arguments: "
                "``velocity_distortion_dimension`` and ``velocity``\n")
            raise KeyError(msg)
        else:
            vel_dist_dim = kwargs['velocity_distortion_dimension']
            velocity = np.copy(kwargs['velocity'])
            apply_distortion = True
    else:
        apply_distortion = False

    if apply_distortion is True:
        try:
            assert vel_dist_dim in ('x', 'y', 'z')
            spatial_distortion = (1. + redshift)*np.copy(velocity)/100./cosmology.efunc(redshift)
            posdict[vel_dist_dim] = np.copy(posdict[vel_dist_dim]) + spatial_distortion
            Lbox = period_dict[vel_dist_dim]
            if Lbox != np.inf:
                posdict[vel_dist_dim] = enforce_periodicity_of_box(
                    posdict[vel_dist_dim], Lbox)
        except AssertionError:
            msg = ("\nInput ``velocity_distortion_dimension`` must be either \n"
                "``'x'``, ``'y'`` or ``'z'``.")
            raise KeyError(msg)

    xout, yout, zout = np.copy(posdict['x']), np.copy(posdict['y']), np.copy(posdict['z'])
    pos = np.vstack([xout, yout, zout]).T

    # Apply a mask, if applicable
    try:
        mask = kwargs['mask']
        return pos[mask]
    except KeyError:
        return pos


def apply_zspace_distortion(true_pos, peculiar_velocity, redshift, cosmology, Lbox=None):
    r""" Apply redshift-space distortions to the comoving simulation coordinate,
    optionally accounting for periodic boundary conditions.

    This function implements the following formula:

    .. math::

        s_{\rm com}^{\rm z-space} = s_{\rm com}^{\rm true} + \frac{1 + z}{H(z)}v_{\rm pec}

    See :ref:`zspace_distortion_derivation` to see where this formula comes from.

    Parameters
    ----------
    true_pos : ndarray
        Array of shape (npts, ) storing the line-of-sight position in comoving Mpc/h.
        In most cases ``true_pos`` is the z-coordinate of the simulation.

    peculiar_velocity : ndarray
        Array of shape (npts, ) storing the peculiar velocity in physical km/s.
        In most cases ``peculiar_velocity`` is the z-velocity of the simulation.

    redshift : float or ndarray
        Float or ndarray of shape (npts, ) storing the redshift of the object.
        If using a single snapshot, this argument is a single float equal to the
        redshift of the snapshot. If using a lightcone, this argument is the
        redshift of each point.

    cosmology : astropy.cosmology.Cosmology
        Cosmology to assume when applying redshift-space distortions,
        e.g., the cosmology of the simulation.

    Lbox : float, optional
        Box length of the simulation so that periodic boundary conditions
        can be applied. Default behavior is None, in which case PBCs will be ignored.

    Returns
    -------
    zspace_pos : ndarray
        Array of shape (npts, ) storing the z-space coordinates in comoving Mpc/h

    Examples
    --------
    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> true_pos = halocat.halo_table['halo_z']
    >>> peculiar_velocity = halocat.halo_table['halo_vz']
    >>> redshift = halocat.redshift
    >>> cosmology = halocat.cosmology
    >>> Lbox = halocat.Lbox[2]
    >>> zspace_zcoord = apply_zspace_distortion(true_pos, peculiar_velocity, redshift, cosmology, Lbox)
    """
    scale_factor = 1./(1. + redshift)
    pos_err = peculiar_velocity/100./cosmology.efunc(redshift)/scale_factor
    zspace_pos = true_pos + pos_err
    if Lbox is not None:
        zspace_pos = enforce_periodicity_of_box(zspace_pos, Lbox)
    return zspace_pos


def cuboid_subvolume_labels(sample, Nsub, Lbox):
    r"""
    Return integer labels indicating which cubical subvolume of a larger cubical volume a
    set of points occupy.

    Parameters
    ----------
    sample : array_like
        Npts x 3 numpy array containing 3-D positions of points.

    Nsub : array_like
        Length-3 numpy array of integers indicating how many times to split the volume
        along each dimension.  If a single integer, N, is supplied, ``Nsub`` is set to
        [N,N,N], and the volume is split along each dimension N times.  The total number
        of subvolumes is given by numpy.prod(Nsub).

    Lbox : array_like
        Length-3 numpy array definging the lengths of the sides of the cubical volume
        that ``sample`` occupies.  If only a single scalar is specified, the volume is assumed
        to be a cube with side-length Lbox

    Returns
    -------
    labels : numpy.array
        (Npts, ) numpy array with integer labels in the range [1,numpy.prod(Nsub)] indicating
        the subvolume each point in ``sample`` occupies.

    N_sub_vol : int
       number of subvolumes.

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

    >>> sample = np.vstack((x,y,z)).T

    Divide the volume into cubes with length 0.25 on a side.

    >>> Nsub = [4,4,4]
    >>> labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    """

    # process inputs and check for consistency
    sample = np.atleast_1d(sample).astype('f8')
    try:
        assert sample.ndim == 2
        assert sample.shape[1] == 3
    except AssertionError:
        msg = ("Input ``sample`` must have shape (Npts, 3)")
        raise TypeError(msg)

    Nsub = np.atleast_1d(Nsub).astype('i4')
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0], Nsub[0], Nsub[0]])
    elif len(Nsub) != 3:
        msg = "Input ``Nsub`` must be a scalar or length-3 sequence"
        raise TypeError(msg)

    Lbox = np.atleast_1d(Lbox).astype('f8')
    if len(Lbox) == 1:
        Lbox = np.array([Lbox[0]]*3)
    elif len(Lbox) != 3:
        msg = "Input ``Lbox`` must be a scalar or length-3 sequence"
        raise TypeError(msg)

    dL = Lbox/Nsub  # length of subvolumes along each dimension
    N_sub_vol = int(np.prod(Nsub))  # total the number of subvolumes
    # create an array of unique integer IDs for each subvolume
    inds = np.arange(1, N_sub_vol+1).reshape(Nsub[0], Nsub[1], Nsub[2])

    # tag each particle with an integer indicating which subvolume it is in
    index = np.floor(sample/dL).astype(int)
    # take care of the case where a point falls on the boundary
    for i in range(3):
        index[:, i] = np.where(index[:, i] == Nsub[i], Nsub[i] - 1, index[:, i])
    index = inds[index[:, 0], index[:, 1], index[:, 2]].astype(int)

    return index, int(N_sub_vol)


def sign_pbc(x1, x2, period=None, equality_fill_val=0., return_pbc_correction=False):
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
    >>> result = sign_pbc(x1, x2, period=Lbox)
    >>> assert result == 1

    >>> result = sign_pbc(x1, x2, period=None)
    >>> assert result == -1

    >>> npts = 100
    >>> x1 = np.random.uniform(0, Lbox, npts)
    >>> x2 = np.random.uniform(0, Lbox, npts)
    >>> result = sign_pbc(x1, x2, period=Lbox)
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


def relative_positions_and_velocities(x1, x2, period=None, **kwargs):
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
    >>> result = relative_positions_and_velocities(x1, x2, period=Lbox)
    >>> assert np.isclose(result, 2)

    >>> result = relative_positions_and_velocities(x1, x2, period=None)
    >>> assert np.isclose(result, -248)

    >>> npts = 100
    >>> x1 = np.random.uniform(0, Lbox, npts)
    >>> x2 = np.random.uniform(0, Lbox, npts)
    >>> result = relative_positions_and_velocities(x1, x2, period=Lbox)

    Now let's frame this result in terms of a physically motivated example.
    Suppose we have a central galaxy with position *xc* and velocity *vc*,
    and a satellite galaxy with position *xs* and velocity *vs*.
    We can calculate the vector pointing from the central to the satellite,
    as well as the satellites's host-centric velocity:

    >>> xcen, vcen = 249.9, 100
    >>> xsat, vsat = 0.1, -300
    >>> xrel, vrel = relative_positions_and_velocities(xsat, xcen, v1=vsat, v2=vcen, period=Lbox)
    >>> assert np.isclose(xrel, +0.2)
    >>> assert np.isclose(vrel, -400)

    >>> xcen, vcen = 0.1, 100
    >>> xsat, vsat = 249.9, -300
    >>> xrel, vrel = relative_positions_and_velocities(xsat, xcen, v1=vsat, v2=vcen, period=Lbox)
    >>> assert np.isclose(xrel, -0.2)
    >>> assert np.isclose(vrel, +400)

    """
    s = sign_pbc(x1, x2, period=period, equality_fill_val=1.)
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
