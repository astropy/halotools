"""
Module containing the `~halotools.mock_observables.radial_pvd_vs_r` function
used to calculate the pairwise radial velocity dispersion
as a function of 3d distance between the pairs.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .pairwise_velocities_helpers import (_pairwise_velocity_stats_process_args,
    _process_radial_bins)

from .velocity_marked_npairs_3d import velocity_marked_npairs_3d

__all__ = ('radial_pvd_vs_r', )
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero


def radial_pvd_vs_r(sample1, velocities1, rbins, sample2=None,
        velocities2=None, period=None, do_auto=True, do_cross=True,
        num_threads=1, max_sample_size=int(1e6),
        approx_cell1_size=None, approx_cell2_size=None):
    """
    Calculate the pairwise velocity dispersion (PVD), :math:`\\sigma_{12}(r)`.

    Example calls to this function appear in the documentation below.

    See also :ref:`galaxy_catalog_analysis_tutorial7`.

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.

    velocities1 : array_like
        len(sample1) array of velocities.

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are
        counted.

    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.

    velocities2 : array_like, optional
        len(sample12) array of velocities.

    period : array_like, optional
        length 3 array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox, Lbox, Lbox].

    do_auto : boolean, optional
        caclulate the auto-pairwise velocities?

    do_cross : boolean, optional
        caclulate the cross-pairwise velocities?

    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.

    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter.
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled
        such that the subsample is equal to max_sample_size.

    Returns
    -------
    sigma_12 : numpy.array
        *len(rbins)-1* length array containing the dispersion of the pairwise velocity,
        :math:`\\sigma_{12}(r)`, computed in each of the bins defined by ``rbins``.

    Notes
    -----
    The pairwise velocity, :math:`v_{12}(r)`, is defined as:

    .. math::
        v_{12}(r) = \\vec{v}_{\\rm 1, pec} \\cdot \\vec{r}_{12}-\\vec{v}_{\\rm 2, pec} \\cdot \\vec{r}_{12}

    where :math:`\\vec{v}_{\\rm 1, pec}` is the peculiar velocity of object 1, and
    :math:`\\vec{r}_{12}` is the radial vector connecting object 1 and 2.

    :math:`\\sigma_{12}(r)` is the standard deviation of this quantity in radial bins.

    Pairs and radial velocities are calculated using
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs`.

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

    >>> coords = np.vstack((x,y,z)).T

    We will do the same to get a random set of peculiar velocities.

    >>> vx = np.random.random(Npts)-0.5
    >>> vy = np.random.random(Npts)-0.5
    >>> vz = np.random.random(Npts)-0.5
    >>> velocities = np.vstack((vx,vy,vz)).T

    >>> rbins = np.logspace(-2,-1,10)
    >>> sigma_12 = radial_pvd_vs_r(coords, velocities, rbins, period=period)

    See also
    ---------
    ref:`galaxy_catalog_analysis_tutorial7`
    """

    #process input arguments
    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads, max_sample_size,
        approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2,\
        period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs =\
        _pairwise_velocity_stats_process_args(*function_args)

    rbins = _process_radial_bins(rbins, period, PBCs)

    #calculate velocity difference scale
    std_v1 = np.sqrt(np.std(velocities1[0, :]))
    std_v2 = np.sqrt(np.std(velocities2[0, :]))

    #build the marks.
    shift1 = np.repeat(std_v1, len(sample1))
    shift2 = np.repeat(std_v2, len(sample2))
    marks1 = np.vstack((sample1.T, velocities1.T, shift1)).T
    marks2 = np.vstack((sample2.T, velocities2.T, shift2)).T

    def marked_pair_counts(sample1, sample2, rbins, period, num_threads,
            do_auto, do_cross, marks1, marks2,
            weight_func_id, _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """

        if do_auto is True:
            D1D1, S1S1, N1N1 = velocity_marked_npairs_3d(
                sample1, sample1, rbins,
                weights1=marks1, weights2=marks1,
                weight_func_id=weight_func_id,
                period=period, num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell1_size)
            D1D1 = np.diff(D1D1)
            S1S1 = np.diff(S1S1)
            N1N1 = np.diff(N1N1)
        else:
            D1D1=None
            D2D2=None
            N1N1=None
            N2N2=None
            S1S1=None
            S2S2=None

        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
            S1S2 = S1S1
            S2S2 = S1S1
        else:
            if do_cross==True:
                D1D2, S1S2, N1N2 = velocity_marked_npairs_3d(
                    sample1, sample2, rbins,
                    weights1=marks1, weights2=marks2,
                    weight_func_id=weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell1_size,
                    approx_cell2_size=approx_cell2_size)
                D1D2 = np.diff(D1D2)
                S1S2 = np.diff(S1S2)
                N1N2 = np.diff(N1N2)
            else:
                D1D2=None
                N1N2=None
                S1S2=None
            if do_auto is True:
                D2D2, S2S2, N2N2 = velocity_marked_npairs_3d(sample2, sample2, rbins,
                    weights1=marks2, weights2=marks2,
                    weight_func_id=weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell2_size,
                    approx_cell2_size=approx_cell2_size)
                D2D2 = np.diff(D2D2)
                S2S2 = np.diff(S2S2)
                N2N2 = np.diff(N2N2)
            else:
                D2D2=None
                N2N2=None

        return D1D1, D1D2, D2D2, S1S1, S1S2, S2S2, N1N1, N1N2, N2N2

    weight_func_id = 12
    V1V1, V1V2, V2V2, S1S1, S1S2, S2S2, N1N1, N1N2, N2N2 = marked_pair_counts(
        sample1, sample2, rbins, period,
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

    #return results
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
