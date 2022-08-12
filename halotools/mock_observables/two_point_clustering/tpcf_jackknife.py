r"""
Module containing the `~halotools.mock_observables.tpcf_jackknife` function used to
calculate the two point correlation function and covariance matrix.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from .tpcf_estimators import _TP_estimator_crossx
from ..pair_counters import npairs_jackknife_3d

from .clustering_helpers import process_optional_input_sample2, verify_tpcf_estimator
from ..mock_observables_helpers import (
    enforce_sample_has_correct_shape,
    get_separation_bins_array,
    get_period,
    get_num_threads,
)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..catalog_analysis_helpers import cuboid_subvolume_labels
from ...custom_exceptions import HalotoolsError

__all__ = ("tpcf_jackknife",)
__author__ = ("Duncan Campbell",)


np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero in e.g. DD/RR


def tpcf_jackknife(
    sample1,
    randoms,
    rbins,
    Nsub=[5, 5, 5],
    sample2=None,
    period=None,
    do_auto=True,
    do_cross=True,
    estimator="Natural",
    num_threads=1,
    seed=None,
):
    r"""
    Calculate the two-point correlation function, :math:`\xi(r)` and the covariance
    matrix, :math:`{C}_{ij}`, between ith and jth radial bin.

    The covariance matrix is calculated using spatial jackknife sampling of the data
    volume.  The spatial samples are defined by splitting the box along each dimension,
    N times, set by the ``Nsub`` argument.

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

    randoms : array_like
        Nran x 3 array containing 3-D positions of randomly distributed points.

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    Nsub : array_like, optional
        Lenght-3 numpy array of number of divisions along each dimension defining
        jackknife sample subvolumes.  If single integer is given, it is assumed to be
        equivalent for each dimension.
        The total number of samples used is then given by
        *numpy.prod(Nsub)*. Default is 5 divisions per dimension.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function. Default is None, in which case only the
        auto-correlation function will be calculated.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity.
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
        Default is 'Natural'.

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
        *len(rbins)-1* length array containing correlation function :math:`\xi(r)`
        computed in each of the radial bins defined by input ``rbins``.

        If ``sample2`` is passed as input, three arrays of length *len(rbins)-1* are
        returned:

        .. math::
            \xi_{11}(r), \xi_{12}(r), \xi_{22}(r)

        The autocorrelation of ``sample1``, the cross-correlation between
        ``sample1`` and ``sample2``, and the autocorrelation of ``sample2``. If
        ``do_auto`` or ``do_cross`` is set to False, the appropriate result(s) is not
        returned.

    cov_matrix(ices) : numpy.ndarray

        *len(rbins)-1* by *len(rbins)-1* ndarray containing the covariance matrix
        :math:`C_{ij}`

        If ``sample2`` is passed as input three ndarrays of shape *len(rbins)-1* by
        *len(rbins)-1* are returned:

        .. math::
            C^{11}_{ij}, C^{12}_{ij}, C^{22}_{ij},

        the associated covariance matrices of
        :math:`\xi_{11}(r), \xi_{12}(r), \xi_{22}(r)`. If ``do_auto`` or ``do_cross``
        is set to False, the appropriate result(s) is not returned.

    Notes
    -----
    The jackknife sampling of pair counts is done internally in
    `~halotools.mock_observables.pair_counters.npairs_jackknife_3d`.

    Pairs are counted such that when 'removing' subvolume :math:`k`, and counting a
    pair in subvolumes :math:`i` and :math:`j`:

    .. math::
        D_i D_j += \left \{
            \begin{array}{ll}
                1.0  & : i \neq k, j \neq k \\
                0.5  & : i \neq k, j=k \\
                0.5  & : i = k, j \neq k \\
                0.0  & : i=j=k \\
            \end{array}
                   \right.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points
    within a periodic cube of box length Lbox = 250 Mpc/h.

    >>> Npts = 1000
    >>> Lbox = 100.

    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    Create some 'randoms' in the same way:

    >>> Nran = Npts*500
    >>> xran = np.random.uniform(0, Lbox, Nran)
    >>> yran = np.random.uniform(0, Lbox, Nran)
    >>> zran = np.random.uniform(0, Lbox, Nran)
    >>> randoms = np.vstack((xran,yran,zran)).T

    Calculate the jackknife covariance matrix by dividing the simulation box
    into 3 samples per dimension (for a total of 3^3 total jackknife samples):

    >>> rbins = np.logspace(0.5, 1.5, 8)
    >>> xi, xi_cov = tpcf_jackknife(coords, randoms, rbins, Nsub=3, period=Lbox)  #  doctest: +SKIP
    """

    # process input parameters
    function_args = (
        sample1,
        randoms,
        rbins,
        Nsub,
        sample2,
        period,
        do_auto,
        do_cross,
        estimator,
        num_threads,
        seed,
    )
    (
        sample1,
        rbins,
        Nsub,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    ) = _tpcf_jackknife_process_args(*function_args)

    # determine box size the data occupies.
    # This is used in determining jackknife samples.
    if PBCs is False:
        sample1, sample2, randoms, Lbox = _enclose_in_box(sample1, sample2, randoms)
    else:
        Lbox = period

    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)

    N1 = len(sample1)
    N2 = len(sample2)
    NR = len(randoms)

    j_index_1, N_sub_vol = cuboid_subvolume_labels(sample1, Nsub, Lbox)
    j_index_2, N_sub_vol = cuboid_subvolume_labels(sample2, Nsub, Lbox)
    j_index_random, N_sub_vol = cuboid_subvolume_labels(randoms, Nsub, Lbox)

    # number of points in each subvolume
    NR_subs = get_subvolume_numbers(j_index_random, N_sub_vol)
    N1_subs = get_subvolume_numbers(j_index_1, N_sub_vol)
    N2_subs = get_subvolume_numbers(j_index_2, N_sub_vol)
    # number of points in each jackknife sample
    N1_subs = N1 - N1_subs
    N2_subs = N2 - N2_subs
    NR_subs = NR - NR_subs

    # calculate all the pair counts
    D1D1, D1D2, D2D2 = jnpair_counts(
        sample1,
        sample2,
        j_index_1,
        j_index_2,
        N_sub_vol,
        rbins,
        period,
        num_threads,
        do_auto,
        do_cross,
        _sample1_is_sample2,
    )

    # pull out the full and sample results
    if do_auto is True:
        D1D1_full = D1D1[0, :]
        D1D1_sub = D1D1[1:, :]
        D2D2_full = D2D2[0, :]
        D2D2_sub = D2D2[1:, :]
    if do_cross is True:
        D1D2_full = D1D2[0, :]
        D1D2_sub = D1D2[1:, :]

    # do random counts
    D1R, RR = jrandom_counts(
        sample1,
        randoms,
        j_index_1,
        j_index_random,
        N_sub_vol,
        rbins,
        period,
        num_threads,
        do_DR,
        do_RR,
    )

    if _sample1_is_sample2:
        D2R = D1R
    else:
        if do_DR is True:
            D2R, RR_dummy = jrandom_counts(
                sample2,
                randoms,
                j_index_2,
                j_index_random,
                N_sub_vol,
                rbins,
                period,
                num_threads,
                do_DR,
                do_RR=False,
            )
        else:
            D2R = None

    if do_DR is True:
        D1R_full = D1R[0, :]
        D1R_sub = D1R[1:, :]
        D2R_full = D2R[0, :]
        D2R_sub = D2R[1:, :]
    else:
        D1R_full = None
        D1R_sub = None
        D2R_full = None
        D2R_sub = None
    if do_RR is True:
        RR_full = RR[0, :]
        RR_sub = RR[1:, :]
    else:
        RR_full = None
        RR_sub = None

    # calculate the correlation function for the full sample
    if do_auto is True:
        xi_11_full = _TP_estimator(
            D1D1_full, D1R_full, RR_full, N1, N1, NR, NR, estimator
        )
        xi_22_full = _TP_estimator(
            D2D2_full, D2R_full, RR_full, N2, N2, NR, NR, estimator
        )
    if do_cross is True:
        xi_12_full = _TP_estimator_crossx(
            D1D2_full, D1R_full, D2R_full, RR_full, N1, N2, NR, NR, estimator
        )

    # calculate the correlation function for the subsamples
    if do_auto is True:
        xi_11_sub = _TP_estimator(
            D1D1_sub, D1R_sub, RR_sub, N1_subs, N1_subs, NR_subs, NR_subs, estimator
        )
        xi_22_sub = _TP_estimator(
            D2D2_sub, D2R_sub, RR_sub, N2_subs, N2_subs, NR_subs, NR_subs, estimator
        )
    if do_cross is True:
        xi_12_sub = _TP_estimator_crossx(
            D1D2_sub,
            D1R_sub,
            D2R_sub,
            RR_sub,
            N1_subs,
            N2_subs,
            NR_subs,
            NR_subs,
            estimator,
        )

    # calculate the covariance matrix
    if do_auto is True:
        xi_11_cov = np.array(np.cov(xi_11_sub.T, bias=True)) * (N_sub_vol - 1.0)
        xi_22_cov = np.array(np.cov(xi_22_sub.T, bias=True)) * (N_sub_vol - 1.0)
    if do_cross is True:
        xi_12_cov = np.array(np.cov(xi_12_sub.T, bias=True)) * (N_sub_vol - 1.0)

    if _sample1_is_sample2:
        return xi_11_full, xi_11_cov
    else:
        if (do_auto is True) & (do_cross is True):
            return xi_11_full, xi_12_full, xi_22_full, xi_11_cov, xi_12_cov, xi_22_cov
        elif do_auto is True:
            return xi_11_full, xi_22_full, xi_11_cov, xi_22_cov
        elif do_cross is True:
            return xi_12_full, xi_12_cov


def _enclose_in_box(data1, data2, data3):
    """
    build axis aligned box which encloses all points.
    shift points so cube's origin is at 0,0,0.
    """

    x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]
    x3, y3, z3 = data3[:, 0], data3[:, 1], data3[:, 2]

    xmin = np.min([np.min(x1), np.min(x2), np.min(x3)])
    ymin = np.min([np.min(y1), np.min(y2), np.min(y3)])
    zmin = np.min([np.min(z1), np.min(z2), np.min(z3)])
    xmax = np.max([np.max(x1), np.max(x2), np.min(x3)])
    ymax = np.max([np.max(y1), np.max(y2), np.min(y3)])
    zmax = np.max([np.max(z1), np.max(z2), np.min(z3)])

    xyzmin = np.min([xmin, ymin, zmin])
    xyzmax = np.max([xmax, ymax, zmax]) - xyzmin

    x1 = x1 - xyzmin
    y1 = y1 - xyzmin
    z1 = z1 - xyzmin
    x2 = x2 - xyzmin
    y2 = y2 - xyzmin
    z2 = z2 - xyzmin
    x3 = x3 - xyzmin
    y3 = y3 - xyzmin
    z3 = z3 - xyzmin

    Lbox = np.array([xyzmax, xyzmax, xyzmax])

    return (
        np.vstack((x1, y1, z1)).T,
        np.vstack((x2, y2, z2)).T,
        np.vstack((x3, y3, z3)).T,
        Lbox,
    )


def get_subvolume_numbers(j_index, N_sub_vol):
    """
    calculate how many points are in each subvolume.
    """

    # there could be subvolumes with no points, and we
    # need every label to be in there at least once. append a vector
    # of the possible labels, and we can subtract 1 later.
    temp = np.hstack((j_index, np.arange(1, N_sub_vol + 1, 1)))

    labels, N = np.unique(temp, return_counts=True)

    N = N - 1  # remove the place holder I added two lines above.

    return N


def jnpair_counts(
    sample1,
    sample2,
    j_index_1,
    j_index_2,
    N_sub_vol,
    rbins,
    period,
    num_threads,
    do_auto,
    do_cross,
    _sample1_is_sample2,
):
    """
    Count jackknife data pairs: DD
    """
    if do_auto is True:
        D1D1 = npairs_jackknife_3d(
            sample1,
            sample1,
            rbins,
            period=period,
            jtags1=j_index_1,
            jtags2=j_index_1,
            N_samples=N_sub_vol,
            num_threads=num_threads,
        )
        D1D1 = np.diff(D1D1, axis=1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_jackknife_3d(
                sample1,
                sample2,
                rbins,
                period=period,
                jtags1=j_index_1,
                jtags2=j_index_2,
                N_samples=N_sub_vol,
                num_threads=num_threads,
            )
            D1D2 = np.diff(D1D2, axis=1)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = npairs_jackknife_3d(
                sample2,
                sample2,
                rbins,
                period=period,
                jtags1=j_index_2,
                jtags2=j_index_2,
                N_samples=N_sub_vol,
                num_threads=num_threads,
            )
            D2D2 = np.diff(D2D2, axis=1)

    return D1D1, D1D2, D2D2


def jrandom_counts(
    sample,
    randoms,
    j_index,
    j_index_randoms,
    N_sub_vol,
    rbins,
    period,
    num_threads,
    do_DR,
    do_RR,
):
    """
    Count jackknife random pairs: DR, RR
    """

    if do_DR is True:
        DR = npairs_jackknife_3d(
            sample,
            randoms,
            rbins,
            period=period,
            jtags1=j_index,
            jtags2=j_index_randoms,
            N_samples=N_sub_vol,
            num_threads=num_threads,
        )
        DR = np.diff(DR, axis=1)
    else:
        DR = None
    if do_RR is True:
        RR = npairs_jackknife_3d(
            randoms,
            randoms,
            rbins,
            period=period,
            jtags1=j_index_randoms,
            jtags2=j_index_randoms,
            N_samples=N_sub_vol,
            num_threads=num_threads,
        )
        RR = np.diff(RR, axis=1)
    else:
        RR = None

    return DR, RR


def _tpcf_jackknife_process_args(
    sample1,
    randoms,
    rbins,
    Nsub,
    sample2,
    period,
    do_auto,
    do_cross,
    estimator,
    num_threads,
    seed,
):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.jackknife_tpcf`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)

    sample2, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1, sample2, do_cross
    )

    period, PBCs = get_period(period)

    # process randoms parameter
    if np.shape(randoms) == (1,):
        N_randoms = randoms[0]
        if PBCs is True:
            with NumpyRNGContext(seed):
                randoms = np.random.random((N_randoms, 3)) * period
        else:
            msg = (
                "\n When no `period` parameter is passed, \n"
                "the user must provide true randoms, and \n"
                "not just the number of randoms desired."
            )
            raise HalotoolsError(msg)

    rbins = get_separation_bins_array(rbins)
    rmax = np.amax(rbins)

    # Process Nsub entry and check for consistency.
    Nsub = np.atleast_1d(Nsub)
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0]] * 3)
    try:
        assert np.all(Nsub < np.inf)
        assert np.all(Nsub > 0)
    except AssertionError:
        msg = "\n Input `Nsub` must be a bounded positive number in all dimensions"
        raise HalotoolsError(msg)

    _enforce_maximum_search_length(rmax, period)

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
        rbins,
        Nsub,
        sample2,
        randoms,
        period,
        do_auto,
        do_cross,
        num_threads,
        _sample1_is_sample2,
        PBCs,
    )
