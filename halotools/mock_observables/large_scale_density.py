# -*- coding: utf-8 -*-

"""
functions to measure large-scale density
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from .pair_counters.double_tree_per_object_pairs import *
from ..custom_exceptions import *
from warnings import warn

from ..utils import convert_to_ndarray
from .large_scale_density_helpers import *
##########################################################################################


__all__ = ('large_scale_density_spherical_volume', 
    'large_scale_density_spherical_annulus')
__author__ = ('Andrew Hearin', )

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def large_scale_density_spherical_volume(sample, tracers, radius, 
    period=None, sample_volume = None, num_threads=1, approx_cell1_size=None, 
    norm_by_mean_density = False):
    """
    Calculate the mean density of the input ``sample`` 
    from an input set of tracer particles. 

    Around each point in the input ``sample``, a sphere of the input ``radius`` 
    is placed and the number of points in the input ``tracers`` is counted, 
    optionally accounting for box periodicity. 
    The `large_scale_density_spherical_volume` returns the mean number density 
    of tracer particles in each such sphere, optionally normalizing this result 
    by the global mean number density of tracer particles in the entire sample volume. 

    Parameters 
    ------------
    sample : array_like 
        Npts x 3 numpy array containing 3-D positions of points.
        See `~halotools.mock_observables.return_xyz_formatted_array` for 
        a convenience function that can be used to transform a set of x, y, z 
        1d arrays into the required form. 

    tracers : array_like 
        Npts x 3 numpy array containing 3-D positions of tracers.

    radius : float 
        Radius of the sphere used to define the volume inside which the 
        number density of tracers is calculated. 

    period : array_like, optional 
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If set to None, PBCs are set to infinity, in which case ``sample_volume`` 
        must be specified. 

    sample_volume : float, optional 
        If period is set to None, you must specify the effective volume of the sample. 

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

    norm_by_mean_density : bool, optional 
        If set to True, the returned number density will be normalized by 
        the global number density of tracer particles averaged across the 
        entire ``sample_volume``. Default is False. 

    Returns 
    --------
    number_density : array_like 
        Length-Npts array of number densities

    Examples 
    ---------
    >>> npts1, npts2 = 100, 200
    >>> sample = np.random.random((npts1, 3))
    >>> tracers = np.random.random((npts2, 3))
    >>> radius = 0.1
    >>> result = large_scale_density_spherical_volume(sample, tracers, radius, period=1)

    """
    sample, tracers, rbins, period, sample_volume, num_threads, approx_cell1_size = (
        _large_scale_density_spherical_volume_process_args(
            sample, tracers, radius, period, sample_volume, num_threads, approx_cell1_size)
        )

    _ = per_object_npairs(sample, tracers, rbins, period = period,
        num_threads = num_threads, approx_cell1_size = approx_cell1_size)
    result = _[:,0]

    environment_volume = (4/3.)*np.pi*radius**3
    number_density = result/environment_volume

    if norm_by_mean_density is True:
        mean_rho = tracers.shape[0]/sample_volume
        return number_density/mean_rho
    else:
        return number_density


def large_scale_density_spherical_annulus(sample, tracers, inner_radius, outer_radius, 
    period=None, sample_volume = None, num_threads=1, approx_cell1_size=None, 
    norm_by_mean_density = False):
    """
    Calculate the mean density of the input ``sample`` 
    from an input set of tracer particles. 

    Around each point in the input ``sample``, a sphere of the input ``radius`` 
    is placed and the number of points in the input ``tracers`` is counted, 
    optionally accounting for box periodicity. 
    The `large_scale_density_spherical_volume` returns the mean number density 
    of tracer particles in each such sphere, optionally normalizing this result 
    by the global mean number density of tracer particles in the entire sample volume. 

    Parameters 
    ------------
    sample : array_like 
        Npts x 3 numpy array containing 3-D positions of points.
        See `~halotools.mock_observables.return_xyz_formatted_array` for 
        a convenience function that can be used to transform a set of x, y, z 
        1d arrays into the required form. 

    tracers : array_like 
        Npts x 3 numpy array containing 3-D positions of tracers.

    inner_radius, outer_radius : float, float
        Radii of the annulus used to define the volume inside which the 
        number density of tracers is calculated. 

    period : array_like, optional 
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If set to None, PBCs are set to infinity, in which case ``sample_volume`` 
        must be specified. 

    sample_volume : float, optional 
        If period is set to None, you must specify the effective volume of the sample. 

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

    norm_by_mean_density : bool, optional 
        If set to True, the returned number density will be normalized by 
        the global number density of tracer particles averaged across the 
        entire ``sample_volume``. Default is False. 

    Returns 
    --------
    number_density : array_like 
        Length-Npts array of number densities

    Examples 
    ---------
    >>> npts1, npts2 = 100, 200
    >>> sample = np.random.random((npts1, 3))
    >>> tracers = np.random.random((npts2, 3))
    >>> inner_radius, outer_radius = 0.1, 0.2
    >>> result = large_scale_density_spherical_annulus(sample, tracers, inner_radius, outer_radius, period=1)

    """
    sample, tracers, rbins, period, sample_volume, num_threads, approx_cell1_size = (
        _large_scale_density_spherical_annulus_process_args(
            sample, tracers, inner_radius, outer_radius, 
            period, sample_volume, num_threads, approx_cell1_size)
        )

    _ = per_object_npairs(sample, tracers, rbins, period = period,
        num_threads = num_threads, approx_cell1_size = approx_cell1_size)
    result = np.diff(_, axis=1)

    environment_volume = (4/3.)*np.pi*(outer_radius**3 - inner_radius**3)
    number_density = result/environment_volume

    if norm_by_mean_density is True:
        mean_rho = tracers.shape[0]/sample_volume
        return number_density/mean_rho
    else:
        return number_density


