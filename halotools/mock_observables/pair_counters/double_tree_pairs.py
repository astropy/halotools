# -*- coding: utf-8 -*-

"""
This module contains functions used to calculate the number of pairs of points
as a function of the separation between the points. Many choices for the
separation variable(s) are available, including 3-D spherical shells,
`~halotools.mock_observables.pair_counters.npairs`, 2+1-D cylindrical shells,
`~halotools.mock_observables.pair_counters.xy_z_npairs`, and separations
:math:`s + \\theta_{\\rm los}` defined by angular & line-of-sight coordinates,
`~halotools.mock_observables.pair_counters.s_mu_npairs`.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

import time
import multiprocessing
from functools import partial
from astropy.extern.six.moves import xrange as range

from .double_tree import FlatRectanguloidDoubleTree
from .double_tree_helpers import *

from .cpairs import *

from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ('s_mu_npairs', )
__author__ = ['Duncan Campbell', 'Andrew Hearin']

##########################################################################


##########################################################################

def s_mu_npairs(data1, data2, s_bins, mu_bins, period = None, 
    verbose = False, num_threads = 1, approx_cell1_size = None, approx_cell2_size = None):
    """
    Function counts the number of pairs of points separated by less than
    radial separation, *s,* and :math:`\\mu\\equiv\\sin(\\theta_{\\rm los})`,
    where :math:`\\theta_{\\rm los}` is the line-of-sight angle
    between points and :math:`s^2 = r_{\\rm parallel}^2 + r_{\\rm perp}^2`.

    Note that if data1 == data2 that the
    `~halotools.mock_observables.s_mu_npairs` function double-counts pairs.
    If your science application requires data1==data2 inputs and also pairs
    to not be double-counted, simply divide the final counts by 2.

    A common variation of pair-counting calculations is to count pairs with
    separations *between* two different distances *r1* and *r2*. You can retrieve
    this information from the `~halotools.mock_observables.s_mu_npairs`
    by taking `numpy.diff` of the returned array.

    See Notes section for further clarification.

    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension
        of the input period.

    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension
        of the input period.

    s_bins : array_like
        numpy array of boundaries defining the radial bins in which pairs are counted.

    mu_bins : array_like
        numpy array of boundaries defining bins in :math:`\\sin(\\theta_{\\rm los})`
        in which the pairs are counted in.
        Note that using the sine is not common convention for
        calculating the two point correlation function (see notes).

    period : array_like, optional
        Length-3 array defining the periodic boundary conditions.
        If only one number is specified, the enclosing volume is assumed to
        be a periodic cube (by far the most common case).
        If period is set to None, the default option,
        PBCs are set to infinity.

    verbose : Boolean, optional
        If True, print out information and progress.

    num_threads : int, optional
        Number of CPU cores to use in the pair counting.
        If ``num_threads`` is set to the string 'max', use all available cores.
        Default is 1 thread for a serial calculation that
        does not open a multiprocessing pool.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by which
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree`
        will apportion the ``data`` points into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use 1/10 of the box size in each dimension,
        which will result in reasonable performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with it when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        See comments for ``approx_cell1_size``.

    Returns
    -------
    num_pairs : array of length len(rbins)
        number of pairs

    Notes
    ------
    The quantity :math:`\\mu` is defined as the :math:`\\sin(\\theta_{\\rm los})`
    and not the conventional :math:`\\cos(\\theta_{\\rm los})`. This is
    because the pair counter has been optimized under the assumption that its
    separation variable (in this case, :math:`\\mu`) *increases*
    as :math:`\\theta_{\\rm los})` increases.

    One final point of clarification concerning double-counting may be in order.
    Suppose data1==data2 and rbins[0]==0. Then the returned value for this bin
    will be len(data1), since each data1 point has distance 0 from itself.

    Returns
    -------
    N_pairs : array_like
        2-d array of length *Num_rp_bins x Num_pi_bins* storing the pair counts in each bin.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1e3, 1e3, 200.
    >>> period = [Lbox, Lbox, Lbox]
    >>> s_bins = np.logspace(-1, 1.25, 15)
    >>> mu_bins = np.linspace(-0.5, 0.5)

    >>> x1 = np.random.uniform(0, Lbox, Npts1)
    >>> y1 = np.random.uniform(0, Lbox, Npts1)
    >>> z1 = np.random.uniform(0, Lbox, Npts1)
    >>> x2 = np.random.uniform(0, Lbox, Npts2)
    >>> y2 = np.random.uniform(0, Lbox, Npts2)
    >>> z2 = np.random.uniform(0, Lbox, Npts2)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> data1 = np.vstack([x1, y1, z1]).T
    >>> data2 = np.vstack([x2, y2, z2]).T

    >>> result = s_mu_npairs(data1, data2, s_bins, mu_bins, period = period)
    """

    #the parameters for this are similar to npairs, except mu_bins needs to be processed.
    # Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, s_bins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )

    xperiod, yperiod, zperiod = period
    rmax = np.max(s_bins)

    #process mu_bins parameter separately
    mu_bins = convert_to_ndarray(mu_bins)
    try:
        assert mu_bins.ndim == 1
        assert len(mu_bins) > 1
        if len(mu_bins) > 2:
            assert array_is_monotonic(mu_bins, strict = True) == 1
    except AssertionError:
        msg = ("\n Input `mu_bins` must be a monotonically increasing \n"
               "1D array with at least two entries")
        raise HalotoolsError(msg)

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        rmax, rmax, rmax, xperiod, yperiod, zperiod, PBCs=PBCs)

    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs

    #create a function to call with only one argument
    engine = partial(_s_mu_npairs_engine, double_tree, s_bins, mu_bins, period, PBCs)

    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        cell1_chunk_list = np.array_split(range(Ncell1), num_threads)
        result = pool.map(engine,cell1_chunk_list)
        pool.close()
        counts = np.sum(np.array(result), axis=0)
    if num_threads == 1:
        counts = engine(range(Ncell1))
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))

    return counts

def _s_mu_npairs_engine(double_tree, s_bins, mu_bins, period, PBCs, cell1_list):
    """
    pair counting engine for s_mu_npairs function.  This code calls a cython function.
    """
    # print("...working on icell1 = %i" % icell1)

    counts = np.zeros((len(s_bins), len(mu_bins)))

    for icell1 in cell1_list:
        #extract the points in the cell
        s1 = double_tree.tree1.slice_array[icell1]
        x_icell1, y_icell1, z_icell1 = (
            double_tree.tree1.x[s1],
            double_tree.tree1.y[s1],
            double_tree.tree1.z[s1])

        xsearch_length = s_bins[-1]
        ysearch_length = s_bins[-1]
        zsearch_length = s_bins[-1]
        adj_cell_generator = double_tree.adjacent_cell_generator(
            icell1, xsearch_length, ysearch_length, zsearch_length)

        for icell2, xshift, yshift, zshift in adj_cell_generator:

            #extract the points in the cell
            s2 = double_tree.tree2.slice_array[icell2]
            x_icell2 = double_tree.tree2.x[s2] + xshift
            y_icell2 = double_tree.tree2.y[s2] + yshift
            z_icell2 = double_tree.tree2.z[s2] + zshift


            #use cython functions to do pair counting
            counts += s_mu_npairs_no_pbc(
                x_icell1, y_icell1, z_icell1,
                x_icell2, y_icell2, z_icell2,
                s_bins, mu_bins)

    return counts


