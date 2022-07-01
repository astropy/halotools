"""
Module containing the `~halotools.mock_observables.void_prob_func`
and `~halotools.mock_observables.underdensity_prob_func` used to calculate void statistics.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.utils.misc import NumpyRNGContext

from ..pair_counters import npairs_per_object_3d

from ...utils.array_utils import array_is_monotonic
from ...custom_exceptions import HalotoolsError


__all__ = ("void_prob_func",)
__author__ = ["Duncan Campbell", "Andrew Hearin"]


np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero in e.g. DD/RR


def void_prob_func(
    sample1,
    rbins,
    n_ran=None,
    random_sphere_centers=None,
    period=None,
    num_threads=1,
    approx_cell1_size=None,
    approx_cellran_size=None,
    seed=None,
):
    """
    Calculate the void probability function (VPF), :math:`P_0(r)`,
    defined as the probability that a random
    sphere of radius *r* contains zero points in the input sample.

    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` argument.

    See also :ref:`galaxy_catalog_analysis_tutorial8`

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rbins : float
        size of spheres to search for neighbors
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    n_ran : int, optional
        integer number of randoms to use to search for voids.
        If ``n_ran`` is not passed, you must pass ``random_sphere_centers``.

    random_sphere_centers : array_like, optional
        Npts x 3 array of randomly selected positions to drop down spheres
        to use to measure the `void_prob_func`. If ``random_sphere_centers``
        is not passed, ``n_ran`` must be passed.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None, PBCs are set to infinity. In this case, it is still necessary
        to drop down randomly placed spheres in order to compute the VPF. To do so,
        the spheres will be dropped inside a cubical box whose sides are defined by
        the smallest/largest coordinate distance of the input ``sample1``.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

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

    approx_cellran_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for randoms.  See comments for
        ``approx_cell1_size`` for details.

    seed : int, optional
        Random number seed used to randomly lay down spheres, if applicable.
        Default is None, in which case results will be stochastic.

    Returns
    -------
    vpf : numpy.array
        *len(rbins)* length array containing the void probability function
        :math:`P_0(r)` computed for each :math:`r` defined by input ``rbins``.

    Notes
    -----
    This function requires the calculation of the number of pairs per randomly placed
    sphere, and thus storage of an array of shape(n_ran,len(rbins)).  This can be a
    memory intensive process as this array becomes large.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 10000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    >>> rbins = np.logspace(-2,-1,20)
    >>> n_ran = 1000
    >>> vpf = void_prob_func(coords, rbins, n_ran=n_ran, period=period)

    See also
    ----------
    :ref:`galaxy_catalog_analysis_tutorial8`

    """
    (
        sample1,
        rbins,
        n_ran,
        random_sphere_centers,
        period,
        num_threads,
        approx_cell1_size,
        approx_cellran_size,
    ) = _void_prob_func_process_args(
        sample1,
        rbins,
        n_ran,
        random_sphere_centers,
        period,
        num_threads,
        approx_cell1_size,
        approx_cellran_size,
        seed,
    )

    result = npairs_per_object_3d(
        random_sphere_centers,
        sample1,
        rbins,
        period=period,
        num_threads=num_threads,
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cellran_size,
    )

    num_empty_spheres = np.array(
        [sum(result[:, i] == 0) for i in range(result.shape[1])]
    )
    return num_empty_spheres / n_ran


def _void_prob_func_process_args(
    sample1,
    rbins,
    n_ran,
    random_sphere_centers,
    period,
    num_threads,
    approx_cell1_size,
    approx_cellran_size,
    seed,
):
    """ """
    sample1 = np.atleast_1d(sample1)

    rbins = np.atleast_1d(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        assert np.min(rbins) > 0
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict=True) == 1
    except AssertionError:
        msg = (
            "\n Input ``rbins`` must be a monotonically increasing \n"
            "1-D array with at least two entries. All entries must be strictly positive."
        )
        raise HalotoolsError(msg)

    if period is None:
        xmin, xmax = np.min(sample1), np.max(sample1)
        ymin, ymax = np.min(sample1), np.max(sample1)
        zmin, zmax = np.min(sample1), np.max(sample1)
    else:
        period = np.atleast_1d(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = "\nInput ``period`` must either be a float or length-3 sequence"
            raise HalotoolsError(msg)
        xmin, xmax = 0.0, float(period[0])
        ymin, ymax = 0.0, float(period[1])
        zmin, zmax = 0.0, float(period[2])

    if n_ran is None:
        if random_sphere_centers is None:
            msg = "You must pass either ``n_ran`` or ``random_sphere_centers``"
            raise HalotoolsError(msg)
        else:
            random_sphere_centers = np.atleast_1d(random_sphere_centers)
            try:
                assert random_sphere_centers.shape[1] == 3
            except AssertionError:
                msg = (
                    "Your input ``random_sphere_centers`` must have shape (Nspheres, 3)"
                )
                raise HalotoolsError(msg)
        n_ran = float(random_sphere_centers.shape[0])
    else:
        if random_sphere_centers is not None:
            msg = "If passing in ``random_sphere_centers``, do not also pass in ``n_ran``."
            raise HalotoolsError(msg)
        else:
            with NumpyRNGContext(seed):
                xran = np.random.uniform(xmin, xmax, n_ran)
                yran = np.random.uniform(ymin, ymax, n_ran)
                zran = np.random.uniform(zmin, zmax, n_ran)
            random_sphere_centers = np.vstack([xran, yran, zran]).T

    return (
        sample1,
        rbins,
        n_ran,
        random_sphere_centers,
        period,
        num_threads,
        approx_cell1_size,
        approx_cellran_size,
    )
