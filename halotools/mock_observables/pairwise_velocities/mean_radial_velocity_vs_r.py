r"""
Module containing the `~halotools.mock_observables.mean_radial_velocity_vs_r` function
used to calculate the pairwise mean radial velocity
as a function of 3d distance between the pairs.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing

from .engines import mean_radial_velocity_vs_r_engine

from ..radial_profiles.radial_profiles_helpers import get_normalized_rbins

from functools import partial

from ..pair_counters.mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from ..pair_counters.mesh_helpers import _enclose_in_box
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_period, get_num_threads)

__all__ = ('mean_radial_velocity_vs_r', )
__author__ = ('Andrew Hearin', 'Duncan Campbell')


def mean_radial_velocity_vs_r(sample1, velocities1,
        rbins_absolute=None, rbins_normalized=None, normalize_rbins_by=None,
        sample2=None, velocities2=None, period=None,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the mean pairwise velocity, :math:`\bar{v}_{12}(r)`.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    See also :ref:`galaxy_catalog_analysis_tutorial6`.

    Parameters
    ----------
    sample1 : array_like
        Numpy array of shape (npts1, 3) containing the 3-D positions of points.

    velocities1 : array_like
        Numpy array of shape (npts1, 3) containing the 3-D velocities.

    rbins_absolute : array_like, optional
        Array of shape (num_rbins+1, ) defining the boundaries of bins in which
        mean radial velocities are computed.

        Either ``rbins_absolute`` must be passed,
        or ``rbins_normalized`` and ``normalize_rbins_by`` must be passed.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rbins_normalized : array_like, optional
        Array of shape (num_rbins+1, ) defining the bin boundaries *x*, where
        :math:`x = r / R_{\rm vir}`, in which mean radial velocity profile is computed.
        The quantity :math:`R_{\rm vir}` can vary from point to point in ``sample1``
        and is passed in via the ``normalize_rbins_by`` argument.
        While scaling by :math:`R_{\rm vir}` is common, you are not limited to this
        normalization choice; in principle you can use the ``rbins_normalized`` and
        ``normalize_rbins_by`` arguments to scale your distances by any length-scale
        associated with points in ``sample1``.

        Default is None, in which case the ``rbins_absolute`` argument must be passed.

    normalize_rbins_by : array_like, optional
        Numpy array of shape (npts1, ) defining how the distance between each pair of points
        will be normalized. For example, if ``normalize_rbins_by`` is defined to be the
        virial radius of each point in ``sample1``, then the input numerical values *x*
        stored in ``rbins_normalized`` will be interpreted as referring to
        bins of :math:`x = r / R_{\rm vir}`. Default is None, in which case
        the input ``rbins_absolute`` argument must be passed instead of
        ``rbins_normalized``.

        Pay special attention to length-units in whatever halo catalog you are using:
        while Halotools-provided catalogs will always have length units
        pre-processed to be Mpc/h, commonly used default settings for ASCII catalogs
        produced by Rockstar return the ``Rvir`` column in kpc/h units,
        but halo centers in Mpc/h units.

    sample2 : array_like, optional
        Numpy array of shape (npts2, 3) containing the 3-D positions of points.

    velocities2 : array_like, optional
        Numpy array of shape (npts2, 3) containing the 3-D velocities.

    period : array_like, optional
        Length-3 array defining periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].
        Default is None, for no PBCs.

    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by how points
        will be apportioned into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use *max(rbins)* in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for `sample2`.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    v_12 : numpy.array
        Array of shape (num_rbins, ) containing the mean pairwise radial velocity

    Notes
    -----
    The pairwise velocity, :math:`v_{12}(r)`, is defined as:

    .. math::

        v_{12}(r) = \vec{v}_{\rm 1, pec} \cdot \vec{r}_{12}-\vec{v}_{\rm 2, pec} \cdot \vec{r}_{12}

    where :math:`\vec{v}_{\rm 1, pec}` is the peculiar velocity of object 1, and
    :math:`\vec{r}_{12}` is the radial vector connecting object 1 and 2.

    :math:`\bar{v}_{12}(r)` is the mean of that quantity calculated in radial bins.

    For radial separation bins in which there are zero pairs, function returns zero.

    Examples
    --------
    For demonstration purposes we will work with
    halos in the `~halotools.sim_manager.FakeSim`. Here we'll just demonstrate
    basic usage, referring to :ref:`galaxy_catalog_analysis_tutorial6` for a
    more detailed demo.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    >>> x = halocat.halo_table['halo_x']
    >>> y = halocat.halo_table['halo_y']
    >>> z = halocat.halo_table['halo_z']

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    We will do the same to get a random set of velocities.

    >>> vx = halocat.halo_table['halo_vx']
    >>> vy = halocat.halo_table['halo_vy']
    >>> vz = halocat.halo_table['halo_vz']
    >>> velocities = np.vstack((vx,vy,vz)).T

    >>> rbins = np.logspace(-1, 1, 10)
    >>> v_12 = mean_radial_velocity_vs_r(sample1, velocities, rbins_absolute=rbins, period=halocat.Lbox)

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial6`

    """
    result = _process_args(sample1, velocities1, sample2, velocities2,
        rbins_absolute, rbins_normalized, normalize_rbins_by,
        period, num_threads, approx_cell1_size, approx_cell2_size)

    sample1, velocities1, sample2, velocities2, max_rbins_absolute, period,\
        num_threads, _sample1_is_sample2, PBCs, \
        approx_cell1_size, approx_cell2_size, rbins_normalized, normalize_rbins_by = result
    x1in, y1in, z1in = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2in, y2in, z2in = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    xperiod, yperiod, zperiod = period
    squared_normalize_rbins_by = normalize_rbins_by*normalize_rbins_by
    search_xlength = max_rbins_absolute
    search_ylength = max_rbins_absolute
    search_zlength = max_rbins_absolute

    #  Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    x1in, y1in, z1in = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2in, y2in, z2in = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    vx1in, vy1in, vz1in = velocities1[:, 0], velocities1[:, 1], velocities1[:, 2]
    vx2in, vy2in, vz2in = velocities2[:, 0], velocities2[:, 1], velocities2[:, 2]

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(mean_radial_velocity_vs_r_engine, double_mesh,
        x1in, y1in, z1in, x2in, y2in, z2in,
        vx1in, vy1in, vz1in, vx2in, vy2in, vz2in,
        squared_normalize_rbins_by, rbins_normalized)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(engine, cell1_tuples))
        counts, vrad_sum = result[:, 0], result[:, 1]
        counts = np.sum(counts, axis=0)
        vrad_sum = np.sum(vrad_sum, axis=0)
        pool.close()
    else:
        counts, vrad_sum = np.array(engine(cell1_tuples[0]))

    counts = np.diff(counts)
    vrad_sum = np.diff(vrad_sum)

    mean_radial_velocity = np.zeros(len(vrad_sum))
    mean_radial_velocity[counts > 0] = vrad_sum[counts > 0]/counts[counts > 0]
    return mean_radial_velocity


def _process_args(sample1, velocities1, sample2, velocities2,
        rbins_absolute, rbins_normalized, normalize_rbins_by,
        period, num_threads, approx_cell1_size, approx_cell2_size):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.pairwise_velocity_stats`.
    """
    rbins_normalized, normalize_rbins_by = get_normalized_rbins(
        rbins_absolute, rbins_normalized, normalize_rbins_by, sample1)

    max_rbins_absolute = np.amax(rbins_normalized)*np.amax(normalize_rbins_by)

    sample1 = enforce_sample_has_correct_shape(sample1)
    velocities1 = np.atleast_1d(velocities1).astype('f4')

    if sample2 is not None:
        sample2 = np.atleast_1d(sample2)
        if velocities2 is None:
            msg = ("\n If `sample2` is passed as an argument, \n"
                   "`velocities2` must also be specified.")
            raise ValueError(msg)
        else:
            velocities2 = np.atleast_1d(velocities2)

        _sample1_is_sample2 = False
        if np.all(sample1.shape == sample2.shape):
            if np.all(sample1 == sample2):
                _sample1_is_sample2 = True
    else:
        sample2 = sample1
        velocities2 = velocities1
        _sample1_is_sample2 = True

    x1 = sample1[:, 0]
    y1 = sample1[:, 1]
    z1 = sample1[:, 2]
    x2 = sample2[:, 0]
    y2 = sample2[:, 1]
    z2 = sample2[:, 2]
    period, PBCs = get_period(period)

    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2,
                min_size=[max_rbins_absolute*3.0, max_rbins_absolute*3.0, max_rbins_absolute*3.0]))
    else:
        PBCs = True
        period = np.atleast_1d(period).astype(float)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise ValueError(msg)

    sample1[:, 0] = x1
    sample1[:, 1] = y1
    sample1[:, 2] = z1
    sample2[:, 0] = x2
    sample2[:, 1] = y2
    sample2[:, 2] = z2

    num_threads = get_num_threads(num_threads)

    if approx_cell1_size is None:
        approx_cell1_size = [max_rbins_absolute, max_rbins_absolute, max_rbins_absolute]
    elif len(np.atleast_1d(approx_cell1_size)) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [max_rbins_absolute, max_rbins_absolute, max_rbins_absolute]
    elif len(np.atleast_1d(approx_cell2_size)) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]

    return sample1, velocities1, sample2, velocities2, max_rbins_absolute, period,\
        num_threads, _sample1_is_sample2, PBCs, approx_cell1_size, approx_cell2_size, \
        rbins_normalized, normalize_rbins_by
