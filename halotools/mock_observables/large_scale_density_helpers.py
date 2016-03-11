# -*- coding: utf-8 -*-

"""
Helper functions used to process the arguments passed 
to functions in the void_stats module. 
"""
import numpy as np 
from ..utils.array_utils import convert_to_ndarray, array_is_monotonic
from ..custom_exceptions import *

__all__ = ('_large_scale_density_spherical_volume_process_args', 
    '_large_scale_density_spherical_annulus_process_args')
__author__ = ('Andrew Hearin', )


def _large_scale_density_spherical_volume_process_args(
    sample, tracers, radius, period, sample_volume, num_threads, approx_cell1_size):
    """
    """
    sample = convert_to_ndarray(sample)
    tracers = convert_to_ndarray(tracers)
    _ = convert_to_ndarray(radius, dt=float)
    rbins = np.append(_, _[0]+0.0001)


    if period is None:
        if sample_volume is None:
            msg = ("If period is None, you must pass in ``sample_volume``.")
            raise HalotoolsError(msg)
        else:
            sample_volume = float(sample_volume)
    else:
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)
        if sample_volume is None:
            sample_volume = period.prod()
        else:
            msg = ("If period is not None, do not pass in sample_volume")
            raise HalotoolsError(msg)

    return sample, tracers, rbins, period, sample_volume, num_threads, approx_cell1_size

def _large_scale_density_spherical_annulus_process_args(
    sample, tracers, inner_radius, outer_radius, 
    period, sample_volume, num_threads, approx_cell1_size):
    """
    """
    sample = convert_to_ndarray(sample)
    tracers = convert_to_ndarray(tracers)

    try:
        assert outer_radius > inner_radius
    except AssertionError:
        msg = ("Input ``outer_radius`` must be larger than input ``inner_radius``")
        raise HalotoolsError(msg)
    rbins = np.array([inner_radius, outer_radius])

    if period is None:
        if sample_volume is None:
            msg = ("If period is None, you must pass in ``sample_volume``.")
            raise HalotoolsError(msg)
        else:
            sample_volume = float(sample_volume)
    else:
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)
        if sample_volume is None:
            sample_volume = period.prod()
        else:
            msg = ("If period is not None, do not pass in sample_volume")
            raise HalotoolsError(msg)

    return sample, tracers, rbins, period, sample_volume, num_threads, approx_cell1_size




