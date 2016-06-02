"""
Module containing the `~halotools.mock_observables.mean_radial_velocity_vs_r` function
used to calculate the pairwise mean radial velocity
as a function of 3d distance between the pairs.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .pairwise_velocities_helpers import _pairwise_velocity_stats_process_args

from .velocity_marked_npairs_3d import velocity_marked_npairs_3d

from ..mock_observables_helpers import get_separation_bins_array

__all__ = ('mean_radial_velocity_vs_r', )
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero


def mean_radial_velocity_vs_r(sample1, velocities1, rbins,
        sample2=None, velocities2=None,
        period=None, do_auto=True, do_cross=True,
        num_threads=1, max_sample_size=int(1e6),
        approx_cell1_size=None, approx_cell2_size=None):
    """
    Calculate the mean pairwise velocity, :math:`\\bar{v}_{12}(r)`.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    See also :ref:`galaxy_catalog_analysis_tutorial6`.

    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing the 3-D positions of points.

    velocities1 : array_like
        N1pts x 3 array containing the 3-D components of the velocities.

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are
        counted.

    sample2 : array_like, optional
        N2pts x 3 array containing the 3-D positions of points.

    velocities2 : array_like, optional
        N2pts x 3 array containing the 3-D components of the velocities.

    period : array_like, optional
        Length-3 array defining  periodic boundary conditions. If only
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
        *len(rbins)-1* length array containing the mean pairwise velocity,
        :math:`\\bar{v}_{12}(r)`, computed in each of the bins defined by ``rbins``.

    Notes
    -----
    The pairwise velocity, :math:`v_{12}(r)`, is defined as:

    .. math::
        v_{12}(r) = \\vec{v}_{\\rm 1, pec} \\cdot \\vec{r}_{12}-\\vec{v}_{\\rm 2, pec} \\cdot \\vec{r}_{12}

    where :math:`\\vec{v}_{\\rm 1, pec}` is the peculiar velocity of object 1, and
    :math:`\\vec{r}_{12}` is the radial vector connecting object 1 and 2.

    :math:`\\bar{v}_{12}(r)` is the mean of that quantity calculated in radial bins.

    Pairs and radial velocities are calculated using
    `~halotools.mock_observables.pair_counters.velocity_marked_npairs`.

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
    >>> v_12 = mean_radial_velocity_vs_r(sample1, velocities, rbins, period=halocat.Lbox)

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial6`

    """

    function_args = (sample1, velocities1, sample2, velocities2, period,
        do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)

    sample1, velocities1, sample2, velocities2, period, do_auto, do_cross,\
        num_threads, _sample1_is_sample2, PBCs = _pairwise_velocity_stats_process_args(*function_args)

    rbins = np.atleast_1d(rbins)

    #create marks for the marked pair counter.
    marks1 = np.vstack((sample1.T, velocities1.T)).T
    marks2 = np.vstack((sample2.T, velocities2.T)).T

    def marked_pair_counts(sample1, sample2, rbins, period, num_threads,
            do_auto, do_cross, marks1, marks2,
            weight_func_id, _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
        """
        Count velocity weighted data pairs.
        """

        if do_auto is True:
            D1D1, dummy, N1N1 = velocity_marked_npairs_3d(
                sample1, sample1, rbins,
                weights1=marks1, weights2=marks1,
                weight_func_id=weight_func_id,
                period=period, num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell1_size)
            D1D1 = np.diff(D1D1)
            N1N1 = np.diff(N1N1)
        else:
            D1D1=None
            D2D2=None
            N1N1=None
            N2N2=None

        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
            N1N2 = N1N1
            N2N2 = N1N1
        else:
            if do_cross is True:
                D1D2, dummy, N1N2 = velocity_marked_npairs_3d(
                    sample1, sample2, rbins,
                    weights1=marks1, weights2=marks2,
                    weight_func_id=weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell1_size,
                    approx_cell2_size=approx_cell2_size)
                D1D2 = np.diff(D1D2)
                N1N2 = np.diff(N1N2)
            else:
                D1D2=None
                N1N2=None
            if do_auto is True:
                D2D2, dummy, N2N2 = velocity_marked_npairs_3d(
                    sample2, sample2, rbins,
                    weights1=marks2, weights2=marks2,
                    weight_func_id=weight_func_id,
                    period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell2_size,
                    approx_cell2_size=approx_cell2_size)
                D2D2 = np.diff(D2D2)
                N2N2 = np.diff(N2N2)
            else:
                D2D2=None
                N2N2=None

        return D1D1, D1D2, D2D2, N1N1, N1N2, N2N2

    #count the sum of radial velocities and number of pairs
    weight_func_id = 11
    V1V1, V1V2, V2V2, N1N1, N1N2, N2N2 =\
        marked_pair_counts(sample1, sample2, rbins, period,
            num_threads, do_auto, do_cross,
            marks1, marks2, weight_func_id,
            _sample1_is_sample2,
            approx_cell1_size, approx_cell2_size)

    #return results: the sum of radial velocities divided by the number of pairs
    if _sample1_is_sample2:
        M_11 = V1V1/N1N1
        return np.where(np.isfinite(M_11), M_11, 0.)
    else:
        if (do_auto is True) & (do_cross is True):
            M_11 = V1V1/N1N1
            M_12 = V1V2/N1N2
            M_22 = V2V2/N2N2
            return (np.where(np.isfinite(M_11), M_11, 0.),
                np.where(np.isfinite(M_12), M_12, 0.), np.where(np.isfinite(M_22), M_22, 0.))
        elif do_cross is True:
            M_12 = V1V2/N1N2
            return np.where(np.isfinite(M_12), M_12, 0.)
        elif (do_auto is True):
            M_11 = V1V1/N1N1
            M_22 = V2V2/N2N2
            return np.where(np.isfinite(M_11), M_11, 0.), np.where(np.isfinite(M_22), M_22, 0.)
