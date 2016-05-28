""" Common functions used when analyzing catalogs of galaxies/halos.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.stats import binned_statistic

from ..empirical_models import enforce_periodicity_of_box

from ..custom_exceptions import HalotoolsError

__all__ = ('mean_y_vs_x', 'return_xyz_formatted_array', 'cuboid_subvolume_labels')
__author__ = ['Andrew Hearin']


def mean_y_vs_x(x, y, error_estimator='error_on_mean', **kwargs):
    """
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
        :math:`\\sigma_{y}/\\sqrt{N}`, where :math:`\\sigma_{y}` is the
        standard deviation of *y* in the bin
        and :math:`\\sqrt{N}` is the counts in each bin.

        If set to ``variance``, function will also return an array storing
        :math:`\\sigma_{y}`.

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


def return_xyz_formatted_array(x, y, z, period=np.inf, **kwargs):
    """ Returns a Numpy array of shape *(Npts, 3)* storing the
    xyz-positions in the format used throughout
    the `~halotools.mock_observables` package.

    See :ref:`mock_obs_pos_formatting` for a tutorial.

    Parameters
    -----------
    x, y, z : sequence of length-Npts arrays
        Units of Mpc assuming h=1, as throughout Halotools.

    velocity : array, optional
        Length-Npts array of velocities in units of km/s
        used to apply peculiar velocity distortions, e.g.,
        :math:`z_{\\rm dist} = z + v/H_{0}`.
        Since Halotools workes exclusively in h=1 units,
        in the above formula :math:`H_{0} = 100 km/s/Mpc`.

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

    mask : array_like, optional
        Boolean mask that can be used to select the positions
        of a subcollection of the galaxies stored in the ``galaxy_table``.

    period : float, optional
        Length of the periodic box. Default is np.inf.

        If period is not np.inf, then after applying peculiar velocity distortions
        the new coordinates will be remapped into the periodic box.

    Returns
    --------
    pos : array_like
        Numpy array with shape *(Npts, 3)*.

    Examples
    ---------
    >>> npts = 100
    >>> Lbox = 250.
    >>> x = np.random.uniform(0, Lbox, npts)
    >>> y = np.random.uniform(0, Lbox, npts)
    >>> z = np.random.uniform(0, Lbox, npts)
    >>> pos = return_xyz_formatted_array(x, y, z, period = Lbox)

    Now we will define an array of random velocities that we will use
    to apply z-space distortions to the z-dimension. For our random velocities
    we'll assume the values are drawn from a Gaussian centered at zero
    using `numpy.random.normal`.

    >>> velocity = np.random.normal(loc=0, scale=100, size=npts)
    >>> pos = return_xyz_formatted_array(x, y, z, period = Lbox, velocity = velocity, velocity_distortion_dimension='z')

    """
    posdict = {'x': np.copy(x), 'y': np.copy(y), 'z': np.copy(z)}

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
            posdict[vel_dist_dim] = np.copy(posdict[vel_dist_dim]) + np.copy(velocity/100.)
            if period != np.inf:
                posdict[vel_dist_dim] = enforce_periodicity_of_box(
                    posdict[vel_dist_dim], period)
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


def cuboid_subvolume_labels(sample, Nsub, Lbox):
    """
    Return integer labels indicating which cubical subvolume of a larger cubical volume a
    set of points occupy.

    Parameters
    ----------
    sample : array_like
        Npts x 3 numpy array containing 3-D positions of points.

    Nsub : array_like
        Length-3 numpy array of integers indicating how many times to split the volume
        along each dimension.  If single integer, N, is supplied, ``Nsub`` is set to
        [N,N,N], and the volume is split along each dimension N times.  The total number
        of subvolumes is then given by numpy.prod(Nsub).

    Lbox : array_like
        Length-3 numpy array definging the lengths of the sides of the cubical volume
        that ``sample`` occupies.  If only a single scalar is specified, the volume is assumed
        to be a cube with side-length Lbox

    Returns
    -------
    labels : numpy.array
        numpy array with integer labels in the range [1,numpy.prod(Nsub)] indicating
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

    #process inputs and check for consistency
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

    #tag each particle with an integer indicating which subvolume it is in
    index = np.floor(sample/dL).astype(int)
    #take care of the case where a point falls on the boundary
    for i in range(3):
        index[:, i] = np.where(index[:, i] == Nsub[i], Nsub[i] - 1, index[:, i])
    index = inds[index[:, 0], index[:, 1], index[:, 2]].astype(int)

    return index, int(N_sub_vol)
