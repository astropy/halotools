r"""
Module containing the `~halotools.mock_observables.alignments.ed_3d_one_two_halo_decomp` function used to
calculate the 1-halo and 2-halo contributions to the ellipticity-direction (ED) correlation functon.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import pi

from .alignment_helpers import process_3d_alignment_args
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_separation_bins_array, get_line_of_sight_bins_array, get_period, get_num_threads)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import positional_marked_npairs_3d, npairs_3d, marked_npairs_3d
from .ed_3d import _ed_3d_process_args

__all__ = ['ed_3d_one_two_halo_decomp']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def ed_3d_one_two_halo_decomp(sample1, orientations1, sample1_host_halo_id,
                              sample2, sample2_host_halo_id, rbins,
                              weights1=None, weights2=None,
                              mask1=None, mask2=None,
                              period=None, num_threads=1,
                              approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the one and two halo componenents of the 3-D ellipticity-direction
    correlation function (ED), :math:`\omega_{\rm 1h}(r)`, and :math:`\omega_{\rm 2h}(r)`.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points with associated orientations.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    orientations1 : array_like
        Npts1 x 3 numpy array containing orientation vectors for each point in ``sample1``.
        these will be normalized if not already.

    sample1_host_halo_id : array_like
        Npts1 length integer array of host halo ids.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.

    sample2_host_halo_id : array_like
        Npts2 length integer array of host halo ids.

    rbins : array_like
        array of boundaries defining the radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    weights1 : array_like, optional
        Npts1 array of weights.  If this parameter is not specified, it is set to numpy.ones(Npts1).

    weights2 : array_like, optional
        Npts2 array of weights.  If this parameter is not specified, it is set to numpy.ones(Npts2).

    mask1 : array_like, optional
        Npts1 boolean array indicating which galaxies in `sample1` contributes to the ED correlation function.
        The default is np.array([True]*Npts1).

    mask2 : array_lile, optional
        Npts2 boolean array indicating which galaxies in `sample2` contributes to the ED correlation function.
        The default is np.array([True]*Npts2).

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity,
        in which case ``randoms`` must be provided.
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

    approx_cell2_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    correlation_functions : numpy.array
        Two *len(rbins)-1* length array containing the correlation function :math:`\omega_{1\rm h}(r)`
        and :math:`\omega_{2\rm h}(r)` computed in each of the bins defined by input ``rbins``.

    Notes
    -----
    The ellipticity-direction correlation function is defined as:

    .. math::
        \omega = \frac{\sum_{i \neq j}w_iw_j|\hat{e}_i \cdot \hat{r}_{ij}|^2}{\sum_{i \neq j} w_iw_j} - \frac{1}{3}

    where e.g. :math:`\hat{e}_i` is the orientation of the :math:`i`-th galaxy, and
    :math:`\hat{r}_{ij}` is the normalized vector in the direction of the :math:`j`-th galaxy
    from the :math:`i`-th galaxy.  :math:`w_i` and :math:`w_j` are the weights associated with
    the :math:`i`-th and :math:`j`-th galaxy. The weights default to 1 if not set.

    Example
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

    We then create a set of random orientation vectors for each point

    >>> random_orientations = np.random.random((Npts,3))

    And a set of random halo ids for each point

    >>> halo_ids = np.random.randint(1, 10, Npts)

    We can the calculate the auto-ED correlation between these points:

    >>> rbins = np.logspace(-1,1,10)
    >>> result = ed_3d_one_two_halo_decomp(sample1, random_orientations,  halo_ids, sample1, halo_ids, rbins, period=Lbox)

    """

    # process arguments
    alignment_args = (sample1, orientations1, None, weights1,
                      sample2, None, None, weights2,
                      None, None, None, None)
    dum = 0.0  # dummy variable to store arguments not needed for this function
    sample1, orientations1, dum, weights1,\
    sample2, dum, dum, weights2,\
    dum, dum, dum, dum = process_3d_alignment_args(*alignment_args)

    function_args = (sample1, rbins, sample2, period, num_threads)
    sample1, rbins, sample2, period, num_threads, PBCs = _ed_3d_process_args(*function_args)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)

    # process mask1
    if mask1 is None:
        mask1 = np.array([True]*N1)
    else:
        mask1 = np.atleast_1d(mask1).astype('bool')
        if np.shape(mask1) != (N1,):
            msg = ('`mask1` is not the correct shape.')
            raise ValueError(msg)
    
    # process mask2
    if mask2 is None:
        mask2 = np.array([True]*N2)
    else:
        mask2 = np.atleast_1d(mask2).astype('bool')
        if np.shape(mask2) != (N2,):
            msg = ('`mask2` is not the correct shape.')
            raise ValueError(msg)

    # process halo ids
    sample1_host_halo_id = np.atleast_1d(sample1_host_halo_id).astype('int')
    sample2_host_halo_id = np.atleast_1d(sample2_host_halo_id).astype('int')
    if np.shape(sample1_host_halo_id) != (N1,):
        msg = ('`sample1_host_halo_id` is not a 1D array of length ``len(samnple1)``.')
        raise ValueError(msg)
    if np.shape(sample2_host_halo_id) != (N2,):
        msg = ('`sample2_host_halo_id` is not a 1D array of length ``len(samnple2)``.')
        raise ValueError(msg)

    marks1 = np.zeros((N1, 5))
    marks1[:,0] = weights1
    marks1[:,1] = orientations1[:,0]
    marks1[:,2] = orientations1[:,1]
    marks1[:,3] = orientations1[:,2]
    marks1[:,4] = sample1_host_halo_id
    marks2 = np.zeros((N2, 2))
    marks2[:,0] = weights2
    marks2[:,1] = sample2_host_halo_id

    marked_counts_1h, counts = positional_marked_npairs_3d(sample1[mask1], sample2[mask2], rbins,
                                period=period, weights1=marks1[mask1], weights2=marks2[mask2],
                                weight_func_id=7, num_threads=num_threads,
                                approx_cell1_size=approx_cell1_size,
                                approx_cell2_size=approx_cell2_size)
    marked_counts_1h = np.diff(marked_counts_1h)

    marked_counts_2h, counts = positional_marked_npairs_3d(sample1[mask1], sample2[mask2], rbins,
                                period=period, weights1=marks1[mask1], weights2=marks2[mask2],
                                weight_func_id=8, num_threads=num_threads,
                                approx_cell1_size=approx_cell1_size,
                                approx_cell2_size=approx_cell2_size)
    marked_counts_2h = np.diff(marked_counts_2h)

    # if no weights, use fast un-weighted pair counter
    if np.all(weights1==1.0) & np.all(weights2==1.0):
        counts = npairs_3d(sample1, sample2, rbins,
                       period=period, num_threads=num_threads,
                       approx_cell1_size=approx_cell1_size,
                       approx_cell2_size=approx_cell2_size)
    else:
        counts = marked_npairs_3d(sample1, sample2, rbins,
                       weights1=weights1, weights2=weights2, weight_func_id=1,
                       period=period, num_threads=num_threads,
                       approx_cell1_size=approx_cell1_size,
                       approx_cell2_size=approx_cell2_size)
    counts = np.diff(counts)

    # get 1-halo and 2-halo pair counts
    marks1 = np.zeros((N1, 2))
    marks1[:,0] = sample1_host_halo_id
    marks1[:,1] = weights1
    marks2 = np.zeros((N2, 2))
    marks2[:,0] = sample2_host_halo_id
    marks2[:,1] = weights2

    counts_1h = marked_npairs_3d(sample1[mask1], sample2[mask2], rbins,
                       weights1=marks1[mask1], weights2=marks2[mask2], weight_func_id=3,
                       period=period, num_threads=num_threads,
                       approx_cell1_size=approx_cell1_size,
                       approx_cell2_size=approx_cell2_size)
    counts_1h = np.diff(counts_1h)

    counts_2h = marked_npairs_3d(sample1[mask1], sample2[mask2], rbins,
                       weights1=marks1[mask1], weights2=marks2[mask2], weight_func_id=4,
                       period=period, num_threads=num_threads,
                       approx_cell1_size=approx_cell1_size,
                       approx_cell2_size=approx_cell2_size)
    counts_2h = np.diff(counts_2h)

    result_1h = marked_counts_1h/counts - 1.0/3.0*(counts_1h/counts)
    result_2h = marked_counts_2h/counts - 1.0/3.0*(counts_2h/counts)

    return result_1h, result_2h


