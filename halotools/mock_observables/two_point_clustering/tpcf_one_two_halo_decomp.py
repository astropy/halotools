r"""
Module containing the `~halotools.mock_observables.tpcf` function used to
calculate the 1-halo, 2-halo decomposition of the
two-point correlation function in 3d.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import gamma

from .clustering_helpers import process_optional_input_sample2, verify_tpcf_estimator

from ..mock_observables_helpers import (
    enforce_sample_has_correct_shape,
    get_separation_bins_array,
    get_period,
    get_num_threads,
)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length

from .tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from .tpcf_estimators import _TP_estimator_crossx
from ..pair_counters import npairs_3d
from ..pair_counters import marked_npairs_3d

from ...custom_exceptions import HalotoolsError

__all__ = ["tpcf_one_two_halo_decomp"]
__author__ = ["Duncan Campbell"]


np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero in e.g. DD/RR


def tpcf_one_two_halo_decomp(
    sample1,
    sample1_host_halo_id,
    rbins,
    sample2=None,
    sample2_host_halo_id=None,
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
    Calculate the real space one-halo and two-halo decomposed two-point correlation
    functions, :math:`\xi^{1h}(r)` and :math:`\xi^{2h}(r)`.

    This returns the correlation function for galaxies which reside in the same halo, and
    those that reside in separate halos, as indicated by a host halo ID.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    See also :ref:`galaxy_catalog_analysis_tutorial2`.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample1_host_halo_id : array_like, optional
        *len(sample1)* integer array of host halo ids.

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function. Default is None, in which case only the
        auto-correlation function will be calculated.

    sample2_host_halo_id : array_like, optional
        *len(sample2)* integer array of host halo ids.

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
    correlation_function(s) : numpy.array
        Two *len(rbins)-1* length arrays containing the one and two halo correlation
        functions, :math:`\xi^{1h}(r)` and :math:`\xi^{2h}(r)`, computed in each of the
        radial bins defined by input ``rbins``.

        .. math::
            1 + \xi(r) \equiv \mathrm{DD} / \mathrm{RR},

        if ``estimator`` is set to 'Natural', where  :math:`\mathrm{DD}` is calculated
        by the pair counter, and :math:`\mathrm{RR}` is counted internally using
        "analytic randoms" if no ``randoms`` are passed as an argument
        (see notes for an explanation).  If a different ``estimator`` is specified, the
        appropiate formula is used.


        If ``sample2`` is passed as input, six arrays of length *len(rbins)-1* are
        returned:

        .. math::
            \xi^{1h}_{11}(r), \ \xi^{2h}_{11}(r),
        .. math::
            \xi^{1h}_{12}(r), \ \xi^{2h}_{12}(r),
        .. math::
            \xi^{1h}_{22}(r), \ \xi^{2h}_{22}(r),

        the autocorrelation of one and two halo autocorrelation of ``sample1``,
        the one and two halo cross-correlation between ``sample1`` and ``sample2``,
        and the one and two halo autocorrelation of ``sample2``.
        If ``do_auto`` or ``do_cross`` is set to False, only the appropriate result(s)
        is returned.

    Examples
    --------
    For demonstration purposes, we'll use the `~halotools.sim_manager.FakeSim` to demonstrate
    how to calculate the 1- and 2-halo term on a set of fake halos.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    >>> x,y,z = halocat.halo_table['halo_x'], halocat.halo_table['halo_y'], halocat.halo_table['halo_z']
    >>> sample1 = np.vstack((x,y,z)).T

    >>> rbins = np.logspace(-2,-1,10)
    >>> host_halo_IDs = halocat.halo_table['halo_hostid']
    >>> xi_1h, xi_2h = tpcf_one_two_halo_decomp(sample1, host_halo_IDs, rbins, period=halocat.Lbox)

    See also
    -----------
    :ref:`galaxy_catalog_analysis_tutorial3`.

    """

    # check input arguments using clustering helper functions
    function_args = (
        sample1,
        sample1_host_halo_id,
        rbins,
        sample2,
        sample2_host_halo_id,
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

    # pass arguments in, and get out processed arguments, plus some control flow variables
    (
        sample1,
        sample1_host_halo_id,
        rbins,
        sample2,
        sample2_host_halo_id,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    ) = _tpcf_one_two_halo_decomp_process_args(*function_args)

    # What needs to be done?
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

    # calculate 1-halo pairs
    weight_func_id = 3
    one_halo_D1D1, one_halo_D1D2, one_halo_D2D2 = marked_pair_counts(
        sample1,
        sample2,
        rbins,
        period,
        num_threads,
        do_auto,
        do_cross,
        sample1_host_halo_id,
        sample2_host_halo_id,
        weight_func_id,
        _sample1_is_sample2,
    )

    # calculate 2-halo pairs
    weight_func_id = 4
    two_halo_D1D1, two_halo_D1D2, two_halo_D2D2 = marked_pair_counts(
        sample1,
        sample2,
        rbins,
        period,
        num_threads,
        do_auto,
        do_cross,
        sample1_host_halo_id,
        sample2_host_halo_id,
        weight_func_id,
        _sample1_is_sample2,
    )

    # count random pairs
    D1R, D2R, RR = random_counts(
        sample1,
        sample2,
        randoms,
        rbins,
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

    # run results through the estimator and return relavent/user specified results.
    if _sample1_is_sample2:
        one_halo_xi_11 = _TP_estimator(
            one_halo_D1D1, D1R, RR, N1, N1, NR, NR, estimator
        )
        two_halo_xi_11 = _TP_estimator(
            two_halo_D1D1, D1R, RR, N1, N1, NR, NR, estimator
        )
        return one_halo_xi_11, two_halo_xi_11
    else:
        if (do_auto is True) & (do_cross is True):
            one_halo_xi_11 = _TP_estimator(
                one_halo_D1D1, D1R, RR, N1, N1, NR, NR, estimator
            )
            one_halo_xi_12 = _TP_estimator_crossx(
                one_halo_D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator
            )
            one_halo_xi_22 = _TP_estimator(
                one_halo_D2D2, D2R, RR, N2, N2, NR, NR, estimator
            )
            two_halo_xi_11 = _TP_estimator(
                two_halo_D1D1, D1R, RR, N1, N1, NR, NR, estimator
            )
            two_halo_xi_12 = _TP_estimator_crossx(
                two_halo_D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator
            )
            two_halo_xi_22 = _TP_estimator(
                two_halo_D2D2, D2R, RR, N2, N2, NR, NR, estimator
            )
            return (
                one_halo_xi_11,
                two_halo_xi_11,
                one_halo_xi_12,
                two_halo_xi_12,
                one_halo_xi_22,
                two_halo_xi_22,
            )
        elif do_cross is True:
            one_halo_xi_12 = _TP_estimator_crossx(
                one_halo_D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator
            )
            two_halo_xi_12 = _TP_estimator_crossx(
                two_halo_D1D2, D1R, D2R, RR, N1, N2, NR, NR, estimator
            )
            return one_halo_xi_12, two_halo_xi_12
        elif do_auto is True:
            one_halo_xi_11 = _TP_estimator(
                one_halo_D1D1, D1R, RR, N1, N1, NR, NR, estimator
            )
            one_halo_xi_22 = _TP_estimator(
                one_halo_D2D2, D2R, RR, N2, N2, NR, NR, estimator
            )
            two_halo_xi_11 = _TP_estimator(
                two_halo_D1D1, D1R, RR, N1, N1, NR, NR, estimator
            )
            two_halo_xi_22 = _TP_estimator(
                two_halo_D2D2, D2R, RR, N2, N2, NR, NR, estimator
            )
            return one_halo_xi_11, two_halo_xi_11, one_halo_xi_22, two_halo_xi_22


def nball_volume(R, k=3):
    """
    Calculate the volume of a n-shpere.
    This is used for the analytical randoms.
    """
    return (np.pi ** (k / 2.0) / gamma(k / 2.0 + 1.0)) * R**k


def random_counts(
    sample1,
    sample2,
    randoms,
    rbins,
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

    # randoms provided, so calculate random pair counts.
    if randoms is not None:
        if do_RR is True:
            RR = npairs_3d(
                randoms,
                randoms,
                rbins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cellran_size,
                approx_cell2_size=approx_cellran_size,
            )
            RR = np.diff(RR)
        else:
            RR = None
        if do_DR is True:
            D1R = npairs_3d(
                sample1,
                randoms,
                rbins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cellran_size,
            )
            D1R = np.diff(D1R)
        else:
            D1R = None
        if _sample1_is_sample2:
            D2R = None
        else:
            if do_DR is True:
                D2R = npairs_3d(
                    sample2,
                    randoms,
                    rbins,
                    period=period,
                    num_threads=num_threads,
                    approx_cell1_size=approx_cell2_size,
                    approx_cell2_size=approx_cellran_size,
                )
                D2R = np.diff(D2R)
            else:
                D2R = None

        return D1R, D2R, RR
    # PBCs and no randoms--calculate randoms analytically.
    elif randoms is None:
        # set the number of randoms equal to the number of points in sample1
        NR = len(sample1)

        # do volume calculations
        v = nball_volume(rbins)  # volume of spheres
        dv = np.diff(v)  # volume of shells
        global_volume = period.prod()  # volume of simulation

        # calculate randoms for sample1
        N1 = np.shape(sample1)[0]  # number of points in sample1
        rho1 = N1 / global_volume  # number density of points
        D1R = (NR) * (dv * rho1)  # random counts are N**2*dv*rho

        # calculate randoms for sample2
        N2 = np.shape(sample2)[0]  # number of points in sample2
        rho2 = N2 / global_volume  # number density of points
        D2R = (NR) * (dv * rho2)  # random counts are N**2*dv*rho

        # calculate the random-random pairs.
        rhor = (NR**2) / global_volume
        RR = dv * rhor

        return D1R, D2R, RR


def marked_pair_counts(
    sample1,
    sample2,
    rbins,
    period,
    num_threads,
    do_auto,
    do_cross,
    marks1,
    marks2,
    weight_func_id,
    _sample1_is_sample2,
):
    """
    Count weighted data pairs.
    """

    # add ones to weights, so returned value is return 1.0*1.0
    marks1 = np.vstack((marks1, np.ones(len(marks1)))).T
    marks2 = np.vstack((marks2, np.ones(len(marks2)))).T

    if do_auto is True:
        D1D1 = marked_npairs_3d(
            sample1,
            sample1,
            rbins,
            weights1=marks1,
            weights2=marks1,
            weight_func_id=weight_func_id,
            period=period,
            num_threads=num_threads,
        )
        D1D1 = np.diff(D1D1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = marked_npairs_3d(
                sample1,
                sample2,
                rbins,
                weights1=marks1,
                weights2=marks2,
                weight_func_id=weight_func_id,
                period=period,
                num_threads=num_threads,
            )
            D1D2 = np.diff(D1D2)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = marked_npairs_3d(
                sample2,
                sample2,
                rbins,
                weights1=marks2,
                weights2=marks2,
                weight_func_id=weight_func_id,
                period=period,
                num_threads=num_threads,
            )
            D2D2 = np.diff(D2D2)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2


def _tpcf_one_two_halo_decomp_process_args(
    sample1,
    sample1_host_halo_id,
    rbins,
    sample2,
    sample2_host_halo_id,
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
    `~halotools.mock_observables.tpcf_one_two_halo_decomp`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)
    sample1_host_halo_id = np.atleast_1d(sample1_host_halo_id).astype(int)

    sample2, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1, sample2, do_cross
    )
    if _sample1_is_sample2 is True:
        sample2_host_halo_id = sample1_host_halo_id
    else:
        if sample2_host_halo_id is None:
            msg = "If passing an input ``sample2``, must also pass sample2_host_halo_id"
            raise ValueError(msg)
        else:
            sample2_host_halo_id = np.atleast_1d(sample2_host_halo_id).astype(int)

    if randoms is not None:
        randoms = np.atleast_1d(randoms)

    # test to see if halo ids are the same length as samples
    if np.shape(sample1_host_halo_id) != (len(sample1),):
        msg = (
            "\n `sample1_host_halo_id` must be a 1-D \n"
            "array the same length as `sample1`."
        )
        raise HalotoolsError(msg)
    if np.shape(sample2_host_halo_id) != (len(sample2),):
        msg = (
            "\n `sample2_host_halo_id` must be a 1-D \n"
            "array the same length as `sample2`."
        )
        raise HalotoolsError(msg)

    rbins = get_separation_bins_array(rbins)
    rmax = np.max(rbins)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length(rmax, period)

    if (randoms is None) & (PBCs is False):
        msg = "\n If no PBCs are specified, randoms must be provided."
        raise HalotoolsError(msg)

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
        sample1_host_halo_id,
        rbins,
        sample2,
        sample2_host_halo_id,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    )
