# -*- coding: utf-8 -*-

"""
functions to measure void statistics
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np

from astropy.extern.six import xrange as range

from .pair_counters.double_tree_per_object_pairs import *
from ..custom_exceptions import *

from ..utils import convert_to_ndarray
from .void_stats_helpers import *
##########################################################################################


__all__=['void_prob_func', 'underdensity_prob_func']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def void_prob_func(sample1, rbins, n_ran=None, random_sphere_centers=None,
    period=None, num_threads=1,
    approx_cell1_size=None, approx_cellran_size=None):
    """
    Calculate the void probability function (VPF), :math:`P_0(r)`,
    defined as the probability that a random
    sphere of radius *r* contains zero points in the input sample.

    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` argument.

    See also :ref:`galaxy_catalog_analysis_tutorial8`

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
        See `~halotools.mock_observables.return_xyz_formatted_array` for
        a convenience function that can be used to transform a set of x, y, z
        1d arrays into the required form.

    rbins : float
        size of spheres to search for neighbors

    n_ran : int, optional
        integer number of randoms to use to search for voids.
        If ``n_ran`` is not passed, you must pass ``random_sphere_centers``.

    random_sphere_centers : array_like, optional
        Npts x 3 array of randomly selected positions to drop down spheres
        to use to measure the `void_prob_func`. If ``random_sphere_centers``
        is not passed, ``n_ran`` must be passed.

    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If set to None, PBCs are set to infinity. Even in this case, it is still necessary
        to drop down randomly placed spheres in order to compute the VPF. To do so,
        the spheres will be dropped inside a cubical box whose sides are defined by
        the smallest/largest coordinate distance of the input ``sample1``.

    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all
        available cores.  num_threads=0 is the default.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by which
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree`
        will apportion the ``sample1`` points into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use *max(rbins)* in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cellran_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for used for randoms.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    vpf : numpy.array
        *len(rbins)* length array containing the void probability function
        :math:`P_0(r)` computed for each :math:`r` defined by input ``rbins``.

    Notes
    -----
    This function requires the calculation of the number of pairs per randomly placed
    sphere, and thus storage of an array of shape(n_ran,len(rbins)).  This can be a
    memory intensive process as this array becomes large.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 10000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    >>> rbins = np.logspace(-2,-1,20)
    >>> n_ran = 1000
    >>> vpf = void_prob_func(coords, rbins, n_ran=n_ran, period=period)

    See also
    ----------
    :ref:`galaxy_catalog_analysis_tutorial8`

    """
    (sample1, rbins, n_ran, random_sphere_centers,
        period, num_threads, approx_cell1_size, approx_cellran_size) = (
        _void_prob_func_process_args(sample1, rbins, n_ran, random_sphere_centers,
            period, num_threads, approx_cell1_size, approx_cellran_size))

    result = per_object_npairs(random_sphere_centers, sample1, rbins, period = period,\
                              num_threads = num_threads,\
                              approx_cell1_size = approx_cell1_size,\
                              approx_cell2_size = approx_cellran_size)

    num_empty_spheres = np.array(
        [sum(result[:,i] == 0) for i in range(result.shape[1])])
    return num_empty_spheres/n_ran


def underdensity_prob_func(sample1, rbins, n_ran=None,
    random_sphere_centers=None, period=None,
    sample_volume = None, u=0.2, num_threads=1,
    approx_cell1_size=None, approx_cellran_size=None):
    """
    Calculate the underdensity probability function (UPF), :math:`P_U(r)`.

    :math:`P_U(r)` is defined as the probability that a randomly placed sphere of size
    :math:`r` encompases a volume with less than a specified number density.

    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` argument.

    See also :ref:`galaxy_catalog_analysis_tutorial8`.

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
        See `~halotools.mock_observables.return_xyz_formatted_array` for
        a convenience function that can be used to transform a set of x, y, z
        1d arrays into the required form.

    rbins : float
        size of spheres to search for neighbors

    n_ran : int, optional
        integer number of randoms to use to search for voids.
        If ``n_ran`` is not passed, you must pass ``random_sphere_centers``.

    random_sphere_centers : array_like, optional
        Npts x 3 array of randomly selected positions to drop down spheres
        to use to measure the `void_prob_func`. If ``random_sphere_centers``
        is not passed, ``n_ran`` must be passed.

    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If set to None, PBCs are set to infinity. Even in this case, it is still necessary
        to drop down randomly placed spheres in order to compute the VPF. To do so,
        the spheres will be dropped inside a cubical box whose sides are defined by
        the smallest/largest coordinate distance of the input ``sample1``.

    sample_volume : float, optional
        If period is set to None, you must specify the effective volume of the sample.

    u : float, optional
        density threshold in units of the mean object density

    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all
        available cores.  num_threads=0 is the default.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by which
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree`
        will apportion the ``sample1`` points into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use *max(rbins)* in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cellran_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for used for randoms.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    upf : numpy.array
        *len(rbins)* length array containing the underdensity probability function
        :math:`P_U(r)` computed for each :math:`r` defined by input ``rbins``.

    Notes
    -----
    This function requires the calculation of the number of pairs per randomly placed
    sphere, and thus storage of an array of shape(n_ran,len(rbins)).  This can be a
    memory intensive process as this array becomes large.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 10000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    >>> rbins = np.logspace(-2,-1,20)
    >>> n_ran = 1000
    >>> upf = underdensity_prob_func(coords, rbins, n_ran=n_ran, period=period, u=0.2)

    See also
    ----------
    :ref:`galaxy_catalog_analysis_tutorial8`
    """
    (sample1, rbins, n_ran, random_sphere_centers, period,
        sample_volume, u, num_threads, approx_cell1_size, approx_cellran_size) = (
        _underdensity_prob_func_process_args(
            sample1, rbins, n_ran, random_sphere_centers,
            period, sample_volume, u,
            num_threads, approx_cell1_size, approx_cellran_size))

    result = per_object_npairs(random_sphere_centers, sample1, rbins, period = period,\
                               num_threads = num_threads,\
                               approx_cell1_size = approx_cell1_size,\
                               approx_cell2_size = approx_cellran_size)

    # calculate the number of galaxies as a
    # function of r that corresponds to the
    # specified under-density
    mean_rho = len(sample1)/sample_volume
    vol = (4.0/3.0)* np.pi * rbins**3
    N_max = mean_rho*vol*u

    num_underdense_spheres = np.array(
        [sum(result[:,i] <= N_max[i]) for i in range(len(N_max))])
    return num_underdense_spheres/n_ran



