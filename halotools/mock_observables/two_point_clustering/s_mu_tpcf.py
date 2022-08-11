r"""
Module containing the `~halotools.mock_observables.s_mu_tpcf` function used to
calculate the redshift-space two-point correlation function , :math:`\xi(s, \mu)`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .clustering_helpers import (
    process_optional_input_sample2,
    verify_tpcf_estimator,
    tpcf_estimator_dd_dr_rr_requirements,
)
from ..mock_observables_helpers import (
    enforce_sample_has_correct_shape,
    get_separation_bins_array,
    get_line_of_sight_bins_array,
    get_period,
    get_num_threads,
)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length

from .tpcf_estimators import _TP_estimator_requirements, _TP_estimator
from .tpcf_estimators import _TP_estimator_crossx
from ..pair_counters import npairs_s_mu

__all__ = ["s_mu_tpcf"]
__author__ = ["Duncan Campbell"]

np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero in e.g. DD/RR


def s_mu_tpcf(
    sample1,
    s_bins,
    mu_bins,
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
    Calculate the redshift space correlation function, :math:`\xi(s, \mu)`

    Divide redshift space into bins of radial separation and angle to to the line-of-sight
    (LOS).  This is a pre-step for calculating correlation function multipoles.

    The first two dimensions (x, y) define the plane for perpendicular distances.
    The third dimension (z) is used for parallel distances.  i.e. x,y positions are on
    the plane of the sky, and z is the radial distance coordinate.  This is the 'distant
    observer' approximation.

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

    s_bins : array_like
        numpy array of shape (num_s_bin_edges, ) storing the :math:`s`
        boundaries defining the bins in which pairs are counted.

    mu_bins : array_like
        numpy array of shape (num_mu_bin_edges, ) storing the
        :math:`\cos(\theta_{\rm LOS})` boundaries defining the bins in
        which pairs are counted. All values must be between [0,1].

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function. Default is None, in which case only the
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
    correlation_function(s) : np.ndarray
        Numpy array of shape (num_s_bin_edges-1, num_mu_bin_edges-1) containing
        the correlation function :math:`\xi(s, \mu)` computed in each of the bins
        defined by input ``s_bins`` and ``mu_bins``.

        .. math::
            1 + \xi(s,\mu) = \mathrm{DD}(s,\mu) / \mathrm{RR}(s,\mu)

        if ``estimator`` is set to 'Natural', where  :math:`\mathrm{DD}(s,\mu)` is
        calculated by the pair counter, and :math:`\mathrm{RR}(s,\mu)` is counted
        internally using "analytic randoms" if ``randoms`` is set to None
        (see notes for further details).


        If ``sample2`` is not None (and not exactly the same as ``sample1``),
        three arrays of shape *len(s_bins)-1* by *len(mu_bins)-1* are returned:

        .. math::
            \xi_{11}(s,\mu), \xi_{12}(s,\mu), \xi_{22}(s,\mu),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and
        ``sample2``, and the autocorrelation of ``sample2``, respectively. If
        ``do_auto`` or ``do_cross`` is set to False, the appropriate result(s) are
        returned.

    Notes
    -----
    Let :math:`\vec{s}` be the radial vector connnecting two points.
    The magnitude, :math:`s`, is:

    .. math::
        s = \sqrt{r_{\parallel}^2+r_{\perp}^2},

    where :math:`r_{\parallel}` is the separation parallel to the LOS
    and :math:`r_{\perp}` is the separation perpednicular to the LOS.  :math:`\mu` is
    the cosine of the angle, :math:`\theta_{\rm LOS}`, between the LOS
    and :math:`\vec{s}`:

    .. math::
        \mu = \cos(\theta_{\rm LOS}) \equiv r_{\parallel}/s.

    Pairs are counted using
    `~halotools.mock_observables.pair_counters.npairs_s_mu`.

    If the ``period`` argument is passed in, the ith coordinate of all points
    must be between 0 and period[i].

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube with Lbox = 250 Mpc/h

    >>> Npts = 1000
    >>> Lbox = 250.

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

    >>> s_bins = np.logspace(-1, 1, 10)
    >>> mu_bins = np.linspace(0, 1, 50)
    >>> xi = s_mu_tpcf(sample1, s_bins, mu_bins, period=Lbox)
    """

    # process arguments
    function_args = (
        sample1,
        s_bins,
        mu_bins,
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
        s_bins,
        mu_bins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    ) = _s_mu_tpcf_process_args(*function_args)

    # what needs to be done?
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

    D1D1, D1D2, D2D2 = pair_counts(
        sample1,
        sample2,
        s_bins,
        mu_bins,
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
        s_bins,
        mu_bins,
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

    # return results.  remember to reverse the final result since
    # the pair counts are done in order of increasing theta_LOS (i.e. decreasing mu)
    if _sample1_is_sample2:
        xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)[:, ::-1]
        return xi_11
    else:
        if (do_auto is True) & (do_cross is True):
            xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)[:, ::-1]
            xi_12 = _TP_estimator_crossx(D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator)[
                :, ::-1
            ]
            xi_22 = _TP_estimator(D2D2, D2R, RR, N2, N2, NR, NR, estimator)[:, ::-1]
            return xi_11, xi_12, xi_22
        elif do_cross is True:
            xi_12 = _TP_estimator_crossx(D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator)[
                :, ::-1
            ]
            return xi_12
        elif do_auto is True:
            xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)[:, ::-1]
            xi_22 = _TP_estimator(D2D2, D2R, RR, N2, N2, NR, NR, estimator)[:, ::-1]
            return xi_11, xi_22


def spherical_sector_volume(s, mu):
    """
    This function is used to calculate analytical randoms.

    Calculate the volume of a spherical sector, used for the analytical randoms.
    https://en.wikipedia.org/wiki/Spherical_sector

    Note that the extra factor of 2 is to get the reflection.
    """
    theta = np.arccos(mu)

    vol = (2.0 * np.pi / 3.0) * np.outer((s**3.0), (1.0 - np.cos(theta))) * 2.0
    return vol


def random_counts(
    sample1,
    sample2,
    randoms,
    s_bins,
    mu_bins,
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
    wedge sectors, which is the correct volume to use for a continious cubic volume
    with PBCs
    """

    # PBCs and randoms.
    if randoms is not None:
        if do_RR is True:
            RR = npairs_s_mu(
                randoms,
                randoms,
                s_bins,
                mu_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cellran_size,
                approx_cell2_size=approx_cellran_size,
            )
            RR = np.diff(np.diff(RR, axis=0), axis=1)
        else:
            RR = None
        if do_DR is True:
            D1R = npairs_s_mu(
                sample1,
                randoms,
                s_bins,
                mu_bins,
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
                D2R = npairs_s_mu(
                    sample2,
                    randoms,
                    s_bins,
                    mu_bins,
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
        mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
        dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
        dv = np.diff(dv, axis=1)  # volume of wedges
        dv = np.diff(dv, axis=0)  # volume of wedge 'pieces'
        global_volume = period.prod()

        # calculate randoms for sample1
        N1 = np.shape(sample1)[0]
        rho1 = N1 / global_volume
        D1R = (N1 - 1.0) * (dv * rho1)  # read note about pair counter

        N2 = np.shape(sample2)[0]
        rho2 = N2 / global_volume
        D2R = (N2 - 1.0) * (dv * rho2)  # read note about pair counter

        # calculate the random-random pairs.
        rhor = NR**2 / global_volume
        RR = dv * rhor

        return D1R, D2R, RR
    else:
        raise ValueError("Un-supported combination of PBCs and randoms provided.")


def pair_counts(
    sample1,
    sample2,
    s_bins,
    mu_bins,
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
    if do_auto is True:
        D1D1 = npairs_s_mu(
            sample1,
            sample1,
            s_bins,
            mu_bins,
            period=period,
            num_threads=num_threads,
            approx_cell1_size=approx_cell1_size,
            approx_cell2_size=approx_cell1_size,
        )
        D1D1 = np.diff(np.diff(D1D1, axis=0), axis=1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_s_mu(
                sample1,
                sample2,
                s_bins,
                mu_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size,
            )
            D1D2 = np.diff(np.diff(D1D2, axis=0), axis=1)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = npairs_s_mu(
                sample2,
                sample2,
                s_bins,
                mu_bins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell2_size,
                approx_cell2_size=approx_cell2_size,
            )
            D2D2 = np.diff(np.diff(D2D2, axis=0), axis=1)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2


def _s_mu_tpcf_process_args(
    sample1,
    s_bins,
    mu_bins,
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
    `~halotools.mock_observables.s_mu_tpcf`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)

    sample2, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1, sample2, do_cross
    )

    if randoms is not None:
        randoms = np.atleast_1d(randoms)

    # process radial bins
    s_bins = get_separation_bins_array(s_bins)
    s_max = np.max(s_bins)

    # process angular bins
    mu_bins = get_line_of_sight_bins_array(mu_bins)

    if (np.min(mu_bins) < 0.0) | (np.max(mu_bins) > 1.0):
        msg = "`mu_bins` must be in the range [0,1]."
        raise ValueError(msg)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length(s_max, period)

    if (randoms is None) & (PBCs is False):
        msg = "\n If no PBCs are specified, randoms must be provided.\n"
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
        s_bins,
        mu_bins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    )
