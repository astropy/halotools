r"""
Module containing the `~halotools.mock_observables.los_pvd_vs_rp` function
used to calculate the pairwise line-of-sight velocity dispersion
as a function of projected distance between the pairs.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .pairwise_velocities_helpers import (_pairwise_velocity_stats_process_args,
    _process_rp_bins)

from .velocity_marked_npairs_xy_z import velocity_marked_npairs_xy_z

__all__ = ('los_pvd_vs_rp', )
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero


def los_pvd_vs_rp(sample1, velocities1, rp_bins, pi_max, sample2=None,
        velocities2=None, period=None, do_auto=True, do_cross=True,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the pairwise line-of-sight (LOS) velocity dispersion (PVD),
    as a function of radial distance from ``sample1`` :math:`\sigma_{z12}(r_p)`.

    Example calls to this function appear in the documentation below.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.

    velocities1 : array_like
        Npts1 x 3 array containing the 3-D components of the velocities.

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_max : float
        maximum LOS separation
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.

    velocities2 : array_like, optional
        Npts2 x 3 array containing the 3-D components of the velocities.

    period : array_like, optional
        length 3 array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].

    do_auto : boolean, optional
        calculate the auto-pairwise velocities?

    do_cross : boolean, optional
        calculate the cross-pairwise velocities?

    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.

    Returns
    -------
    sigma : numpy.array or tuple(numpy.arrays)
        Each numpy.array is a *len(rbins)-1* length array containing the dispersion
        of the pairwise velocity, :math:`\sigma_{12}(r)`, computed in each of the bins
        defined by ``rbins``.
        If sample2 is None, returns :math:`\sigma_{11}(r)`
        If ``do_auto`` and ``do_cross`` are True, returns (:math:`\sigma_{11}(r)`, :math:`\sigma_{12}(r)`, :math:`\sigma_{22}(r)`)
        If only ``do_auto`` is True, returns (:math:`\sigma_{11}(r)`, :math:`\sigma_{22}(r)`)
        If only ``do_cross`` is True, returns :math:`\sigma_{12}(r)`

    Notes
    -----
    The pairwise LOS velocity, :math:`v_{z12}(r)`, is defined as:

    .. math::
        v_{z12} = |\vec{v}_{\rm 1, pec}\cdot \hat{z}-\vec{v}_{\rm 2, pec}\cdot\hat{z}|

    where :math:`\vec{v}_{\rm 1, pec}` is the peculiar velocity of object 1, and
    :math:`\hat{z}` is the unit-z vector.

    :math:`\sigma_{z12}(r_p)` is the standard deviation of this quantity in
    projected radial bins.

    Pairs and radial velocities are calculated using
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs_xy_z`.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> from halotools.mock_observables import los_pvd_vs_rp
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    We will do the same to get a random set of peculiar velocities.

    >>> vx = np.random.random(Npts)
    >>> vy = np.random.random(Npts)
    >>> vz = np.random.random(Npts)
    >>> velocities = np.vstack((vx,vy,vz)).T

    >>> rp_bins = np.logspace(-2,-1,10)
    >>> pi_max = 0.3
    >>> sigmaz_12 = los_pvd_vs_rp(coords, velocities, rp_bins, pi_max, period=period)

    >>> x2 = np.random.random(Npts)
    >>> y2 = np.random.random(Npts)
    >>> z2 = np.random.random(Npts)
    >>> coords2 = np.vstack((x2,y2,z2)).T

    >>> vx2 = np.random.random(Npts)
    >>> vy2 = np.random.random(Npts)
    >>> vz2 = np.random.random(Npts)
    >>> velocities2 = np.vstack((vx2,vy2,vz2)).T

    >>> sigmaz_12 = los_pvd_vs_rp(coords, velocities, rp_bins, pi_max, period=period, sample2=coords2, velocities2=velocities2)



    """

    # process input arguments
    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads,
        approx_cell1_size, approx_cell2_size, None)
    sample1, velocities1, sample2, velocities2,\
        period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs =\
        _pairwise_velocity_stats_process_args(*function_args)

    rp_bins, pi_max = _process_rp_bins(rp_bins, pi_max, period, PBCs)
    pi_bins = np.array([0.0, pi_max])

    # calculate velocity difference scale
    std_v1 = np.sqrt(np.std(velocities1[2, :]))
    std_v2 = np.sqrt(np.std(velocities2[2, :]))

    # build the marks.
    shift1 = np.repeat(std_v1, len(sample1))
    shift2 = np.repeat(std_v2, len(sample2))
    marks1 = np.vstack((sample1.T, velocities1.T, shift1)).T
    marks2 = np.vstack((sample2.T, velocities2.T, shift2)).T

    def marked_pair_counts(sample1, sample2, rp_bins, pi_bins, period, num_threads,
            do_auto, do_cross, marks1, marks2,
            weight_func_id, _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """

        if do_auto is True:
            D1D1, S1S1, N1N1 = velocity_marked_npairs_xy_z(
                sample1, sample1, rp_bins, pi_bins,
                weights1=marks1, weights2=marks1, weight_func_id=weight_func_id,
                period=period, num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell1_size)
            D1D1 = np.diff(D1D1, axis=1)[:, 0]
            D1D1 = np.diff(D1D1)
            S1S1 = np.diff(S1S1, axis=1)[:, 0]
            S1S1 = np.diff(S1S1)
            N1N1 = np.diff(N1N1, axis=1)[:, 0]
            N1N1 = np.diff(N1N1)
        else:
            D1D1 = None
            D2D2 = None
            N1N1 = None
            N2N2 = None
            S1S1 = None
            S2S2 = None

        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
            S1S2 = S1S1
            S2S2 = S1S1
        else:
            if do_cross is True:
                D1D2, S1S2, N1N2 = velocity_marked_npairs_xy_z(
                    sample1, sample2, rp_bins, pi_bins,
                    weights1=marks1, weights2=marks2,
                    weight_func_id=weight_func_id, period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell1_size,
                    approx_cell2_size=approx_cell2_size)
                D1D2 = np.diff(D1D2, axis=1)[:, 0]
                D1D2 = np.diff(D1D2)
                S1S2 = np.diff(S1S2, axis=1)[:, 0]
                S1S2 = np.diff(S1S2)
                N1N2 = np.diff(N1N2, axis=1)[:, 0]
                N1N2 = np.diff(N1N2)
            else:
                D1D2 = None
                N1N2 = None
                S1S2 = None
            if do_auto is True:
                D2D2, S2S2, N2N2 = velocity_marked_npairs_xy_z(
                    sample2, sample2, rp_bins, pi_bins,
                    weights1=marks2, weights2=marks2,
                    weight_func_id=weight_func_id, period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell2_size,
                    approx_cell2_size=approx_cell2_size)
                D2D2 = np.diff(D2D2, axis=1)[:, 0]
                D2D2 = np.diff(D2D2)
                S2S2 = np.diff(S2S2, axis=1)[:, 0]
                S2S2 = np.diff(S2S2)
                N2N2 = np.diff(N2N2, axis=1)[:, 0]
                N2N2 = np.diff(N2N2)
            else:
                D2D2 = None
                N2N2 = None

        return D1D1, D1D2, D2D2, S1S1, S1S2, S2S2, N1N1, N1N2, N2N2

    weight_func_id = 4
    V1V1, V1V2, V2V2, S1S1, S1S2, S2S2, N1N1, N1N2, N2N2 = marked_pair_counts(
        sample1, sample2, rp_bins, pi_bins, period,
        num_threads, do_auto, do_cross,
        marks1, marks2, weight_func_id,
        _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size)

    def _shifted_std(N, sum_x, sum_x_sqr):
        """
        calculate the variance
        """
        variance = (sum_x_sqr - (sum_x * sum_x)/N)/(N - 1)
        return np.sqrt(variance)

    # return results
    if _sample1_is_sample2:
        sigma_11 = _shifted_std(N1N1, V1V1, S1S1)
        return np.where(np.isfinite(sigma_11), sigma_11, 0.)
    else:
        if (do_auto is True) & (do_cross is True):
            sigma_11 = _shifted_std(N1N1, V1V1, S1S1)
            sigma_12 = _shifted_std(N1N2, V1V2, S1S2)
            sigma_22 = _shifted_std(N2N2, V2V2, S2S2)
            return (np.where(np.isfinite(sigma_11), sigma_11, 0.),
                np.where(np.isfinite(sigma_12), sigma_12, 0.),
                np.where(np.isfinite(sigma_22), sigma_22, 0.))
        elif (do_cross is True):
            sigma_12 = _shifted_std(N1N2, V1V2, S1S2)
            return np.where(np.isfinite(sigma_12), sigma_12, 0.)
        elif (do_auto is True):
            sigma_11 = _shifted_std(N1N1, V1V1, S1S1)
            sigma_22 = _shifted_std(N2N2, V2V2, S2S2)
            return (np.where(np.isfinite(sigma_11), sigma_11, 0.),
                np.where(np.isfinite(sigma_22), sigma_22, 0.))
