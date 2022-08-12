r"""
Module containing the `~halotools.mock_observables.tpcf` function used to
calculate the two-point correlation function in 3d (aka galaxy clustering).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import gamma
from warnings import warn

from .clustering_helpers import (
    process_optional_input_sample2,
    verify_tpcf_estimator,
    tpcf_estimator_dd_dr_rr_requirements,
)
from .tpcf_estimators import _TP_estimator
from .tpcf_estimators import _TP_estimator_crossx

from ..mock_observables_helpers import (
    enforce_sample_has_correct_shape,
    get_separation_bins_array,
    get_period,
    get_num_threads,
)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import npairs_3d

from ...custom_exceptions import HalotoolsError

##########################################################################################


__all__ = ["tpcf"]
__author__ = ["Duncan Campbell"]

np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero in e.g. DD/RR


def _random_counts(
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
    Internal function used to random pairs during the calculation of the tpcf.
    There are two high level branches:
        1. w/ or wo/ PBCs and randoms.
        2. PBCs and analytical randoms
    There is also control flow governing whether RR and DR pairs are counted,
    as not all estimators need one or the other.

    Analytical counts are N**2*dv*rho, where dv is the volume of the spherical
    shells, which is the correct volume to use for a continious cubical volume with PBCs.
    """

    def nball_volume(R, k=3):
        """
        Calculate the volume of a n-shpere.
        This is used for the analytical randoms.
        """
        return (np.pi ** (k / 2.0) / gamma(k / 2.0 + 1.0)) * R**k

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


def _pair_counts(
    sample1,
    sample2,
    rbins,
    period,
    num_threads,
    do_auto,
    do_cross,
    _sample1_is_sample2,
    approx_cell1_size,
    approx_cell2_size,
):
    r"""
    Internal function used calculate DD-pairs during the calculation of the tpcf.
    """
    if do_auto is True:
        D1D1 = npairs_3d(
            sample1,
            sample1,
            rbins,
            period=period,
            num_threads=num_threads,
            approx_cell1_size=approx_cell1_size,
            approx_cell2_size=approx_cell1_size,
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
            D1D2 = npairs_3d(
                sample1,
                sample2,
                rbins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size,
            )
            D1D2 = np.diff(D1D2)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = npairs_3d(
                sample2,
                sample2,
                rbins,
                period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell2_size,
                approx_cell2_size=approx_cell2_size,
            )
            D2D2 = np.diff(D2D2)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2


def tpcf(
    sample1,
    rbins,
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
    RR_precomputed=None,
    NR_precomputed=None,
    seed=None,
):
    r"""
    Calculate the real space two-point correlation function, :math:`\xi(r)`.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    See also :ref:`galaxy_catalog_analysis_tutorial2` for example usage on a
    mock galaxy catalog.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are counted.
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

    RR_precomputed : array_like, optional
        Array storing the number of RR-counts calculated in advance during
        a pre-processing phase. Must have the same length as *len(rbins)*.
        If the ``RR_precomputed`` argument is provided,
        you must also provide the ``NR_precomputed`` argument.
        Default is None.

    NR_precomputed : int, optional
        Number of points in the random sample used to calculate ``RR_precomputed``.
        If the ``NR_precomputed`` argument is provided,
        you must also provide the ``RR_precomputed`` argument.
        Default is None.

    seed : int, optional
        Random number seed used to randomly downsample data, if applicable.
        Default is None, in which case downsampling will be stochastic.

    Returns
    -------
    correlation_function(s) : numpy.array
        *len(rbins)-1* length array containing the correlation function :math:`\xi(r)`
        computed in each of the bins defined by input ``rbins``.

        .. math::
            1 + \xi(r) \equiv \mathrm{DD}(r) / \mathrm{RR}(r),

        If ``estimator`` is set to 'Natural'.  :math:`\mathrm{DD}(r)` is the number
        of sample pairs with separations equal to :math:`r`, calculated by the pair
        counter.  :math:`\mathrm{RR}(r)` is the number of random pairs with separations
        equal to :math:`r`, and is counted internally using "analytic randoms" if
        ``randoms`` is set to None (see notes for an explanation), otherwise it is
        calculated using the pair counter.

        If ``sample2`` is passed as input
        (and if ``sample2`` is not exactly the same as ``sample1``),
        then three arrays of length *len(rbins)-1* are returned:

        .. math::
            \xi_{11}(r), \xi_{12}(r), \xi_{22}(r),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and
        ``sample2``, and the autocorrelation of ``sample2``, respectively.
        If ``do_auto`` or ``do_cross`` is set to False,
        the appropriate sequence of results is returned.

    Notes
    -----
    For a higher-performance implementation of the tpcf function written in C,
    see the Corrfunc code written by Manodeep Sinha, available at
    https://github.com/manodeep/Corrfunc.

    Examples
    --------
    For demonstration purposes we calculate the `tpcf` for halos in the
    `~halotools.sim_manager.FakeSim`.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    >>> x = halocat.halo_table['halo_x']
    >>> y = halocat.halo_table['halo_y']
    >>> z = halocat.halo_table['halo_z']

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper
    behavior around `numpy.vstack` such as placing points into redshift-space.

    >>> rbins = np.logspace(-1, 1, 10)
    >>> xi = tpcf(sample1, rbins, period=halocat.Lbox)

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial2`
    """

    # check input arguments using clustering helper functions
    function_args = (
        sample1,
        rbins,
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
        RR_precomputed,
        NR_precomputed,
        seed,
    )

    # pass arguments in, and get out processed arguments, plus some control flow variables
    (
        sample1,
        rbins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
        RR_precomputed,
        NR_precomputed,
    ) = _tpcf_process_args(*function_args)

    # What needs to be done?
    do_DD, do_DR, do_RR = tpcf_estimator_dd_dr_rr_requirements[estimator]
    if RR_precomputed is not None:
        # overwrite do_RR as necessary
        do_RR = False

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    if randoms is not None:
        NR = len(randoms)
    else:
        # set the number of randoms equal to the number of points in sample1
        # this is arbitrarily set, but must remain consistent!
        if NR_precomputed is not None:
            NR = NR_precomputed
        else:
            NR = N1

    # count data pairs
    D1D1, D1D2, D2D2 = _pair_counts(
        sample1,
        sample2,
        rbins,
        period,
        num_threads,
        do_auto,
        do_cross,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
    )

    # count random pairs
    D1R, D2R, RR = _random_counts(
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
    if RR_precomputed is not None:
        RR = RR_precomputed

    # run results through the estimator and return relavent/user specified results.
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


def _tpcf_process_args(
    sample1,
    rbins,
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
    RR_precomputed,
    NR_precomputed,
    seed,
):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.tpcf`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)
    sample2, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1, sample2, do_cross
    )

    if randoms is not None:
        randoms = np.atleast_1d(randoms)

    rbins = get_separation_bins_array(rbins)
    rmax = np.amax(rbins)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length(rmax, period)

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

    if (RR_precomputed is not None) | (NR_precomputed is not None):
        try:
            assert ((RR_precomputed is not None) & (NR_precomputed is not None)) is True
        except AssertionError:
            msg = (
                "\nYou must either provide both "
                "``RR_precomputed`` and ``NR_precomputed`` arguments, or neither\n"
            )
            raise HalotoolsError(msg)
        # At this point, we have been provided *both* RR_precomputed *and* NR_precomputed

        try:
            assert len(RR_precomputed) == len(rbins) - 1
        except AssertionError:
            msg = "\nLength of ``RR_precomputed`` must match length of ``rbins``\n"
            raise HalotoolsError(msg)

        if np.any(RR_precomputed == 0):
            msg = (
                "RR_precomputed has radial bin(s) which contain no pairs. \n"
                "Consider increasing the number of randoms, or using larger bins."
            )
            warn(msg)

        try:
            assert len(randoms) == NR_precomputed
        except AssertionError:
            msg = (
                "If passing in randoms and also NR_precomputed, \n"
                "the value of NR_precomputed must agree with the number of randoms\n"
            )
            raise HalotoolsError(msg)

    assert np.all(rbins > 0.0), "All values of input ``rbins`` must be positive"

    return (
        sample1,
        rbins,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
        RR_precomputed,
        NR_precomputed,
    )
