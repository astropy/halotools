""" Module containing the `~halotools.mock_observables.counts_in_cylinders` function
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing
from functools import partial

from .engines import counts_in_cylinders_engine

from ..mock_observables_helpers import get_num_threads, get_period
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from ..pair_counters.mesh_helpers import (
    _set_approximate_cell_sizes,
    _cell1_parallelization_indices,
    _enclose_in_box,
    _enforce_maximum_search_length,
)

from ...utils.array_utils import custom_len

__author__ = ("Andrew Hearin",)

__all__ = ("counts_in_cylinders",)


def counts_in_cylinders(
    sample1,
    sample2,
    proj_search_radius,
    cylinder_half_length,
    period=None,
    verbose=False,
    num_threads=1,
    approx_cell1_size=None,
    approx_cell2_size=None,
    return_indexes=False,
    condition=None,
    condition_args=(),
):
    """
    Function counts the number of points in ``sample2`` separated by a xy-distance
    *r* and z-distance *z* from each point in ``sample1``,
    where *r* and *z* are defined by the input ``proj_search_radius``
    and ``cylinder_half_length``, respectively.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        Npts2 x 3 array containing 3-D positions of points. If this is None, an
        autocorrelation on sample1 will be done instead.

    proj_search_radius : array_like
        Length-Npts1 array defining the xy-distance around each point in ``sample1``
        to search for points in ``sample2``.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    cylinder_half_length : array_like
        Length-Npts1 array defining the z-distance around each point in ``sample1``
        to search for points in ``sample2``. Thus the *total* length of the cylinder
        placed around each point in ``sample1`` will be *twice* the corresponding
        value stored in the input ``cylinder_half_length``.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like, optional
        Length-3 array defining the periodic boundary conditions.
        If only one number is specified, the enclosing volume is assumed to
        be a periodic cube (by far the most common case).
        If period is set to None, the default option,
        PBCs are set to infinity.

    verbose : Boolean, optional
        If True, print out information and progress.

    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed
        using the python ``multiprocessing`` module. Default is 1 for a purely serial
        calculation, in which case a multiprocessing Pool object will
        never be instantiated. A string 'max' may be used to indicate that
        the pair counters should use all available cores on the machine.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by how points
        will be apportioned into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use Lbox/10 in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for
        ``approx_cell1_size`` for details.

    return_indexes: bool, optional
        If true, return both counts and the indexes of the pairs.

    condition : str, optional
        Require a condition to be met for a pair to be counted.
        See options below:
        None | "always_true": Count all pairs in cylinder

        "mass_frac":
            Only count pairs which satisfy lim[0] < mass2/mass1 < lim[1]

    condition_args : tuple, optional
        Arguments passed to the condition constructor
        None | "always_true": condition_args will be ignored

        "mass_frac":
            -mass1 (array of mass of sample 1; required)
            -mass2 (array of mass of sample 2; required)
            -lim (tuple of min,max; required)
            -lower_equality (bool to use lim[0] <= m2/m1; optional)
            -upper_equality (bool to use m2/m1 <= lim[1]; optional)

    Returns
    -------
    num_pairs : array_like
        Numpy array of length Npts1 storing the numbers of points in ``sample2``
        inside the cylinder surrounding each point in ``sample1``.

    indexes : array_like, optional
        If ``return_indexes`` is true, return a structured array of length
        num_pairs with the indexes of the pairs. Column ``i1`` is the index in
        ``sample1`` at the center of the cylinder and column ``i2`` is the index
        in ``sample2`` that is contained in the cylinder.

    Examples
    --------
    For illustration purposes, we'll create some fake data and call the pair counter.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    In this first example, we'll demonstrate how to calculate the number of
    low-mass host halos are in cylinders of *fixed length* surrounding high-mass halos.

    >>> host_halo_mask = halocat.halo_table['halo_upid'] == -1
    >>> host_halos = halocat.halo_table[host_halo_mask]
    >>> high_mass_mask = host_halos['halo_mvir'] >= 5e13
    >>> high_mass_hosts = host_halos[high_mass_mask]
    >>> low_mass_mask = host_halos['halo_mvir'] <= 1e12
    >>> low_mass_hosts = host_halos[low_mass_mask]

    >>> x1, y1, z1 = high_mass_hosts['halo_x'], high_mass_hosts['halo_y'], high_mass_hosts['halo_z']
    >>> x2, y2, z2 = low_mass_hosts['halo_x'], low_mass_hosts['halo_y'], low_mass_hosts['halo_z']

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack([x1, y1, z1]).T
    >>> sample2 = np.vstack([x2, y2, z2]).T

    Now let's drop a cylinder of radius 200 kpc/h and half-length 250 kpc/h around
    each high-mass host halo, and for each high-mass host we'll count the number of
    low-mass halos falling within that cylinder:

    >>> period = halocat.Lbox
    >>> proj_search_radius, cylinder_half_length = 0.2, 0.25
    >>> result = counts_in_cylinders(sample1, sample2, proj_search_radius, cylinder_half_length, period=period)

    For example usage of the `~halotools.mock_observables.counts_in_cylinders` function
    on a realistic galaxy catalog that makes use of the variable search length feature,
    see the :ref:`calculating_counts_in_cells` tutorial.


    """

    # Process the inputs with the helper function
    result = _counts_in_cylinders_process_args(
        sample1,
        sample2,
        proj_search_radius,
        cylinder_half_length,
        period,
        verbose,
        num_threads,
        approx_cell1_size,
        approx_cell2_size,
        return_indexes,
    )
    (
        x1in,
        y1in,
        z1in,
        x2in,
        y2in,
        z2in,
        proj_search_radius,
        cylinder_half_length,
    ) = result[0:8]
    period, num_threads, PBCs, approx_cell1_size, approx_cell2_size, autocorr = result[
        8:
    ]
    xperiod, yperiod, zperiod = period

    rp_max = np.max(proj_search_radius)
    pi_max = np.max(cylinder_half_length)
    search_xlength, search_ylength, search_zlength = rp_max, rp_max, pi_max

    # Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = _set_approximate_cell_sizes(
        approx_cell1_size, approx_cell2_size, period
    )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(
        x1in,
        y1in,
        z1in,
        x2in,
        y2in,
        z2in,
        approx_x1cell_size,
        approx_y1cell_size,
        approx_z1cell_size,
        approx_x2cell_size,
        approx_y2cell_size,
        approx_z2cell_size,
        search_xlength,
        search_ylength,
        search_zlength,
        xperiod,
        yperiod,
        zperiod,
        PBCs,
    )

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(
        counts_in_cylinders_engine,
        double_mesh,
        x1in,
        y1in,
        z1in,
        x2in,
        y2in,
        z2in,
        proj_search_radius,
        cylinder_half_length,
        return_indexes,
        condition,
        condition_args,
    )

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads
    )

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        pool.close()
        if return_indexes:
            counts = np.sum([res[0] for res in result], axis=0)
            indexes = np.concatenate([res[1] for res in result])
        else:
            counts = np.sum(result, axis=0)
    else:
        result = engine(cell1_tuples[0])
        if return_indexes:
            counts = result[0]
            indexes = result[1]
        else:
            counts = result

    # All pairs will match with themselves. We don't want this if we are doing an autocorr
    if autocorr:
        counts -= 1
        if return_indexes:
            indexes = indexes[indexes["i1"] != indexes["i2"]]

    if return_indexes:
        return counts, indexes
    return counts


def _counts_in_cylinders_process_args(
    sample1,
    sample2,
    proj_search_radius,
    cylinder_half_length,
    period,
    verbose,
    num_threads,
    approx_cell1_size,
    approx_cell2_size,
    return_indexes,
):
    """"""
    num_threads = get_num_threads(num_threads)

    autocorr = False
    if sample2 is None:
        sample2, autocorr = sample1, True

    # The engine expects position arrays to be double-precision
    sample1 = np.asarray(sample1, dtype=np.float64)
    sample2 = np.asarray(sample2, dtype=np.float64)

    # Passively enforce that we are working with ndarrays
    x1 = sample1[:, 0]
    y1 = sample1[:, 1]
    z1 = sample1[:, 2]
    x2 = sample2[:, 0]
    y2 = sample2[:, 1]
    z2 = sample2[:, 2]

    if return_indexes and ((len(x1) > 2**32) or (len(x2) > 2**32)):
        msg = (
            "Return indexes uses a uint32 and so can only handle inputs shorter than "
            + "2^32 (~4 Billion). If you are affected by this please raise an Issue on "
            + "https://github.com/astropy/halotools.\n"
        )
        raise ValueError(msg)

    proj_search_radius = np.atleast_1d(proj_search_radius).astype("f8")
    if len(proj_search_radius) == 1:
        proj_search_radius = np.zeros_like(x1) + proj_search_radius[0]
    elif len(proj_search_radius) == len(x1):
        pass
    else:
        msg = "Input ``proj_search_radius`` must be a scalar or length-Npts1 array"
        raise ValueError(msg)
    max_rp_max = np.max(proj_search_radius)

    cylinder_half_length = np.atleast_1d(cylinder_half_length).astype("f8")
    if len(cylinder_half_length) == 1:
        cylinder_half_length = np.zeros_like(x1) + cylinder_half_length[0]
    elif len(cylinder_half_length) == len(x1):
        pass
    else:
        msg = "Input ``cylinder_half_length`` must be a scalar or length-Npts1 array"
        raise ValueError(msg)
    max_pi_max = np.max(cylinder_half_length)

    period, PBCs = get_period(period)
    # At this point, period may still be set to None,
    # in which case we must remap our points inside the smallest enclosing cube
    # and set ``period`` equal to this cube size.
    if period is None:
        x1, y1, z1, x2, y2, z2, period = _enclose_in_box(
            sample1[:, 0],
            sample1[:, 2],
            sample1[:, 2],
            sample2[:, 0],
            sample2[:, 2],
            sample2[:, 2],
            min_size=[max_rp_max * 3.0, max_rp_max * 3.0, max_pi_max * 3.0],
        )
        x1 = sample1[:, 0]
        y1 = sample1[:, 1]
        z1 = sample1[:, 2]
        x2 = sample2[:, 0]
        y2 = sample2[:, 1]
        z2 = sample2[:, 2]
    else:
        x1 = sample1[:, 0]
        y1 = sample1[:, 1]
        z1 = sample1[:, 2]
        x2 = sample2[:, 0]
        y2 = sample2[:, 1]
        z2 = sample2[:, 2]

    _enforce_maximum_search_length(max_rp_max, period[0])
    _enforce_maximum_search_length(max_rp_max, period[1])
    _enforce_maximum_search_length(max_pi_max, period[2])

    if approx_cell1_size is None:
        approx_cell1_size = [max_rp_max, max_rp_max, max_pi_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [max_rp_max, max_rp_max, max_pi_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]

    return (
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        proj_search_radius,
        cylinder_half_length,
        period,
        num_threads,
        PBCs,
        approx_cell1_size,
        approx_cell2_size,
        autocorr,
    )
