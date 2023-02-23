r"""
Module containing the `~halotools.mock_observables.alignments.ee_3d` function used to
calculate the ellipticity-ellipticity (EE) correlation functon
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import pi

from .alignment_helpers import process_3d_alignment_args
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_separation_bins_array, get_line_of_sight_bins_array, get_period, get_num_threads)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import marked_npairs_3d, npairs_3d, marked_npairs_3d

__all__ = ['ee_3d']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def ee_3d(sample1, orientations1, sample2, orientations2, rbins, weights1=None, weights2=None,
        period=None, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the 3-D ellipticity-ellipticity correlation function (EE), :math:`\eta(r)`.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points with associated orientations.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    orientations1 : array_like
        Npts1 x 2 numpy array containing projected orientation vectors for each point in ``sample1``.
        these will be normalized if not already.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.

    orientations2 : array_like
        Npts1 x 3 numpy array containing orientation vectors for each point in ``sample2``.
        these will be normalized if not already.

    rbins : array_like
        array of boundaries defining the radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.
    
     weights1 : array_like, optional
        Npts1 array of weghts.  If this parameter is not specified, it is set to numpy.ones(Npts1).

    weights2 : array_like, optional
        Npts2 array of weghts.  If this parameter is not specified, it is set to numpy.ones(Npts2).

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
    correlation_function : numpy.array
        *len(rbins)-1* length array containing the correlation function :math:`\eta(r)`
        computed in each of the bins defined by input ``rbins``.

    Notes
    -----
    The ellipticity-ellipticity correlation function is defined as:

    .. math::
        \eta = \frac{\sum_{i \neq j}w_iw_j|\hat{e}_i \cdot \hat{e}_j|^2}{\sum_{i \neq j} w_i w_j} - \frac{1}{3}

    where e.g. :math:`\hat{e}_i` is the orientation of the :math:`i`-th galaxy. 
    :math:`w_i` and :math:`w_j` are the weights associated with the :math:`i`-th 
    and :math:`j`-th galaxy. The weights default to 1 if not set.

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

    We can the calculate the auto-EE correlation between these points:

    >>> rbins = np.logspace(-1,1,10)
    >>> result = ee_3d(sample1, random_orientations, sample1, random_orientations, rbins, period=Lbox)

    """

    # process arguments
    alignment_args = (sample1, orientations1, None, weights1,
                      sample2, orientations2, None, weights2,
                      None, None, None, None)
    dum = 0.0  # dummy variable to store arguments not needed for this function
    sample1, orientations1, dum, weights1,\
    sample2, orientations2, dum, weights2,\
    dum, dum, dum, dum = process_3d_alignment_args(*alignment_args)

    function_args = (sample1, rbins, sample2, period, num_threads)
    sample1, rbins, sample2, period, num_threads, PBCs = _ee_3d_process_args(*function_args)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)

    marks1 = np.zeros((N1, 4))
    marks1[:,0] = weights1
    marks1[:,1] = orientations1[:,0]
    marks1[:,2] = orientations1[:,1]
    marks1[:,3] = orientations1[:,2]
    marks2 = np.zeros((N2, 4))
    marks2[:,0] = weights2
    marks2[:,1] = orientations2[:,0]
    marks2[:,2] = orientations2[:,1]
    marks2[:,3] = orientations2[:,2]
    marked_counts = marked_npairs_3d(sample1, sample2, rbins,
                                     period=period, weights1=marks1, weights2=marks2,
                                     weight_func_id=13, num_threads=num_threads,
                                     approx_cell1_size=approx_cell1_size, approx_cell2_size=approx_cell2_size)
    marked_counts = np.diff(marked_counts)

    # if no weights, use fast un-weihgted pair counter
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

    return marked_counts/counts - 1.0/3.0


def _ee_3d_process_args(sample1, rbins, sample2, period, num_threads):
    r"""
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.alignments.ee_3d`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)

    rbins = get_separation_bins_array(rbins)
    rmax = np.amax(rbins)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length([rmax, rmax, rmax], period)

    num_threads = get_num_threads(num_threads)

    return sample1, rbins, sample2, period, num_threads, PBCs

