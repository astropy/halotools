r"""
Module containing the `~halotools.mock_observables.rp_pi_tpcf` function used to
calculate the redshift-space two-point correlation function in 3d, :math:`\xi(r_{p}, \pi)`
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import pi

from .clustering_helpers import process_optional_input_sample2, verify_tpcf_estimator
from .tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from .tpcf_estimators import _TP_estimator_crossx

from ..mock_observables_helpers import (
    enforce_sample_has_correct_shape,
    get_separation_bins_array,
    get_line_of_sight_bins_array,
    get_period,
    get_num_threads,
)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import npairs_xy_z


__all__ = ["rp_pi_tpcf"]
__author__ = ["Duncan Campbell"]


np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero in e.g. DD/RR


def rp_pi_tpcf(
    sample1,
    rp_bins,
    pi_bins,
    sample2=None,
    randoms=None,
    period=None,
    do_auto=True,
    do_cross=True,
    estimator="Natural",
    num_threads=1,
    approx_cell1_size=None,
    approx_cell2_size=None,
    approx_cellran_size=None,
    seed=None,
):
    r"""
    Calculate the redshift space correlation function, :math:`\xi(r_{p}, \pi)`

    Calculate the correlation function as a function of pair separation perpendicular to
    the line-of-sight (LOS) and parallel to the LOS.

    The first two dimensions (x, y) define the plane for perpendicular distances.
    The third dimension (z) is used for parallel distances,  i.e. x,y positions are on
    the plane of the sky, and z is the radial distance coordinate.
    This is the 'distant observer' approximation.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_bins : array_like
        array of boundaries defining the p radial bins parallel to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function.
        Default is None, in which case only the
        auto-correlation function will be calculated.

    randoms : array_like, optional
        Nran x 3 array containing 3-D positions of randomly distributed points.
        If no randoms are provided (the default option),
        calculation of the tpcf can proceed using analytical randoms
        (only valid for periodic boundary conditions).

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity,
        in which case ``randoms`` must be provided.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    do_auto : boolean, optional
        Boolean determines whether the auto-correlation function will
        be calculated and returned. Default is True.

    do_cross : boolean, optional
        Boolean determines whether the cross-correlation function will
        be calculated and returned. Only relevant when ``sample2`` is also provided.
        Default is True for the case where ``sample2`` is provided, otherwise False.

    estimator : string, optional
        Statistical estimator for the tpcf.
        Options are 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
        Default is ``Natural``.

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

    approx_cellran_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for randoms.  See comments for
        ``approx_cell1_size`` for details.

    seed : int, optional
        Random number seed used to randomly downsample data, if applicable.
        Default is None, in which case downsampling will be stochastic.

    Returns
    -------
    correlation_function(s) : numpy.ndarray
        *len(rp_bins)-1* by *len(pi_bins)-1* ndarray containing the correlation function
        :math:`\xi(r_p, \pi)` computed in each of the bins defined by input ``rp_bins``
        and ``pi_bins``.

        .. math::
            1 + \xi(r_{p},\pi) = \mathrm{DD}r_{p},\pi) / \mathrm{RR}r_{p},\pi)

        if ``estimator`` is set to 'Natural', where  :math:`\mathrm{DD}(r_{p},\pi)`
        is calculated by the pair counter, and :math:`\mathrm{RR}(r_{p},\pi)` is counted
        internally using "analytic randoms" if ``randoms`` is set to None
        (see notes for further details).

        If ``sample2`` is passed as input (and not exactly the same as ``sample1``),
        three arrays of shape *len(rp_bins)-1* by *len(pi_bins)-1* are returned:

        .. math::
            \xi_{11}(r_{p},\pi), \xi_{12}(r_{p},\pi), \xi_{22}(r_{p},\pi),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and
        ``sample2``, and the autocorrelation of ``sample2``, respectively. If
        ``do_auto`` or ``do_cross`` is set to False, the appropriate result(s) are
        returned.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube of Lbox = 250 Mpc/h.

    >>> Npts = 1000
    >>> Lbox = 250

    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper
    behavior around `numpy.vstack` such as placing points into redshift-space.

    >>> rp_bins = np.logspace(-1,1,10)
    >>> pi_bins = np.logspace(-1,1,10)
    >>> xi = rp_pi_tpcf(sample1, rp_bins, pi_bins, period=Lbox)

    """

    function_args = (
        sample1,
        rp_bins,
        pi_bins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        estimator,
        num_threads,
        approx_cell1_size,
        approx_cell2_size,
        approx_cellran_size,
        seed,
    )

    (
        sample1,
        rp_bins,
        pi_bins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    ) = _rp_pi_tpcf_process_args(*function_args)

    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    if randoms is not None:
        NR = len(randoms)
    else:
        # set the number of randoms equal to the number of points in sample1
        # this is arbitrarily set, but must remain consistent!
        NR = N1

    # count pairs
    D1D1, D1D2, D2D2 = pair_counts(
        sample1,
        sample2,
        rp_bins,
        pi_bins,
        period,
        num_threads,
        do_auto,
        do_cross,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
    )

    D1R, D2R, RR = random_counts(
        sample1,
        sample2,
        randoms,
        rp_bins,
        pi_bins,
        period,
        PBCs,
        num_threads,
        do_RR,
        do_DR,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
        approx_cellran_size,
    )

    if _sample1_is_sample2:
        xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)
        return xi_11
    else:
        if (do_auto is True) & (do_cross is True):
            xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)
            xi_12 = _TP_estimator_crossx(D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator)
            xi_22 = _TP_estimator(D2D2, D2R, RR, N2, N2, NR, NR, estimator)
            return xi_11, xi_12, xi_22
        elif do_cross is True:
            xi_12 = _TP_estimator_crossx(D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator)
            return xi_12
        elif do_auto is True:
            xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)
            xi_22 = _TP_estimator(D2D2, D2R, RR, N2, N2, NR, NR, estimator)
            return xi_11, xi_22


def pair_counts(
    sample1,
    sample2,
    rp_bins,
    pi_bins,
    period,
    num_threads,
    do_auto,
    do_cross,
    _sample1_is_sample2,
    approx_cell1_size,
    approx_cell2_size,
):
    """
    Count data pairs.
    """
    D1D1 = npairs_xy_z(
        sample1,
        sample1,
        rp_bins,
        pi_bins,
        period=period,
        num_threads=num_threads,
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell1_size,
    )
    D1D1 = np.diff(np.diff(D1D1, axis=0), axis=1)
    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_xy_z(
                sample1,
                sample2,
                rp_bins,
                pi_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size,
            )
            D1D2 = np.diff(np.diff(D1D2, axis=0), axis=1)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = npairs_xy_z(
                sample2,
                sample2,
                rp_bins,
                pi_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell2_size,
                approx_cell2_size=approx_cell2_size,
            )
            D2D2 = np.diff(np.diff(D2D2, axis=0), axis=1)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2


def cylinder_volume(R, h):
    """
    Calculate the volume of a cylinder(s), used for the analytical randoms.
    """
    return pi * np.outer(R**2.0, h)


def random_counts(
    sample1,
    sample2,
    randoms,
    rp_bins,
    pi_bins,
    period,
    PBCs,
    num_threads,
    do_RR,
    do_DR,
    _sample1_is_sample2,
    approx_cell1_size,
    approx_cell2_size,
    approx_cellran_size,
):
    r"""
    Count random pairs.  There are two high level branches:
        1. w/ or wo/ PBCs and randoms.
        2. PBCs and analytical randoms
    There are also logical bits to do RR and DR pair counts, as not all estimators
    need one or the other, and not doing these can save a lot of calculation.

    Analytical counts are N**2*dv*rho, where dv can is the volume of the spherical
    shells, which is the correct volume to use for a continious cubic volume with PBCs
    """

    # No PBCs, randoms must have been provided.
    if randoms is not None:
        if do_RR is True:
            RR = npairs_xy_z(
                randoms,
                randoms,
                rp_bins,
                pi_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cellran_size,
                approx_cell2_size=approx_cellran_size,
            )
            RR = np.diff(np.diff(RR, axis=0), axis=1)
        else:
            RR = None
        if do_DR is True:
            D1R = npairs_xy_z(
                sample1,
                randoms,
                rp_bins,
                pi_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cellran_size,
            )
            D1R = np.diff(np.diff(D1R, axis=0), axis=1)
        else:
            D1R = None
        if _sample1_is_sample2:  # calculating the cross-correlation
            D2R = None
        else:
            if do_DR is True:
                D2R = npairs_xy_z(
                    sample2,
                    randoms,
                    rp_bins,
                    pi_bins,
                    period=period,
                    num_threads=num_threads,
                    approx_cell1_size=approx_cell2_size,
                    approx_cell2_size=approx_cellran_size,
                )
                D2R = np.diff(np.diff(D2R, axis=0), axis=1)
            else:
                D2R = None

        return D1R, D2R, RR
    # PBCs and no randoms--calculate randoms analytically.
    elif randoms is None:

        # set the number of randoms equal to the number of points in sample1
        NR = len(sample1)

        # do volume calculations
        v = cylinder_volume(rp_bins, 2.0 * pi_bins)  # volume of spheres
        dv = np.diff(np.diff(v, axis=0), axis=1)  # volume of annuli
        global_volume = period.prod()

        # calculate randoms for sample1
        N1 = np.shape(sample1)[0]
        rho1 = N1 / global_volume
        D1R = (N1) * (dv * rho1)  # read note about pair counter

        # calculate randoms for sample2
        N2 = np.shape(sample2)[0]
        rho2 = N2 / global_volume
        D2R = N2 * (dv * rho2)  # read note about pair counter

        # calculate the random-random pairs.
        rhor = NR**2 / global_volume
        RR = dv * rhor  # RR is only the RR for the cross-correlation.

        return D1R, D2R, RR


def _rp_pi_tpcf_process_args(
    sample1,
    rp_bins,
    pi_bins,
    sample2,
    randoms,
    period,
    do_auto,
    do_cross,
    estimator,
    num_threads,
    approx_cell1_size,
    approx_cell2_size,
    approx_cellran_size,
    seed,
):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.redshift_space_tpcf`.
    """
    sample1 = enforce_sample_has_correct_shape(sample1)
    sample2, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1, sample2, do_cross
    )

    if randoms is not None:
        randoms = np.atleast_1d(randoms)

    rp_bins = get_separation_bins_array(rp_bins)
    rp_max = np.amax(rp_bins)

    pi_bins = get_line_of_sight_bins_array(pi_bins)
    pi_max = np.amax(pi_bins)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length([rp_max, rp_max, pi_max], period)

    if (randoms is None) & (PBCs is False):
        msg = "If no PBCs are specified, randoms must be provided.\n"
        raise ValueError(msg)

    try:
        assert do_auto == bool(do_auto)
        assert do_cross == bool(do_cross)
    except:
        msg = "`do_auto` and `do_cross` keywords must be boolean-valued."
        raise ValueError(msg)

    num_threads = get_num_threads(num_threads)

    verify_tpcf_estimator(estimator)

    return (
        sample1,
        rp_bins,
        pi_bins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    )
